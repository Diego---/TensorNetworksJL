using LinearAlgebra

"""
    MPO{T}

Open-boundary Matrix Product Operator (MPO) with element type `T`.

The MPO consists of one rank-4 tensor for each physical site. The tensor
at site `n` follows the index convention

    (left bond, top physical index, bottom physical index, right bond)

or, in tensor notation,

    O⁽ⁿ⁾[μₙ₋₁, tₙ, bₙ, μₙ].

The top physical index `tₙ` is the output index and the bottom physical
index `bₙ` is the input index, so locally the tensor represents matrix
elements proportional to |tₙ⟩⟨bₙ|.

For an MPO with `N` sites, the boundary bond dimensions are fixed to one,

    χ₀ = χ_N = 1,

so that the tensor dimensions are

    (1, dᵤ, d_b, χ₁),
    (χ₁, dᵤ, d_b, χ₂),
    ...
    (χₙ₋₂, dᵤ, d_b, χₙ₋₁),
    (χₙ₋₁, dᵤ, d_b, 1),

where `dᵤ` is the dimension of the top (output) physical index,
`d_b` is the dimension of the bottom (input) physical index,
which in general will be of the same size as these operators usually
don't move to a different Hilbert space, and `χᵢ` is the dimension
of the virtual bond index `μᵢ`.

# Fields
- `tensors::Vector{Tensor{T,4}}`: Site tensors ordered from left to right
  along the physical chain.
"""
struct MPO{T}
    tensors::Vector{Tensor{T,4}}
end

function MPO(tensors::Vector{Tensor{T,4}}) where {T}
    # An MPO must contain at least one physical site.
    isempty(tensors) &&
        throw(ArgumentError("An MPO must contain at least one tensor."))

    # Open boundary conditions require trivial outer bond dimensions.
    size(first(tensors), 1) == 1 ||
        throw(DimensionMismatch(
            "The left boundary bond dimension must be 1."
        ))

    size(last(tensors), 4) == 1 ||
        throw(DimensionMismatch(
            "The right boundary bond dimension must be 1."
        ))

    # Every site must have the same output physical dimension.
    d_out = size(first(tensors), 2)
    for (n, O) in enumerate(tensors)
        size(O, 2) == d_out ||
            throw(DimensionMismatch(
                "Output physical dimension at site $n does not match the rest of the MPO."
            ))
    end

    # Every site must have the same input physical dimension.
    d_in = size(first(tensors), 3)
    for (n, O) in enumerate(tensors)
        size(O, 3) == d_in ||
            throw(DimensionMismatch(
                "Input physical dimension at site $n does not match the rest of the MPO."
            ))
    end

    # Adjacent virtual bond dimensions must agree.
    for n in 1:length(tensors)-1
        right_dim = size(tensors[n], 4)
        left_dim = size(tensors[n + 1], 1)

        right_dim == left_dim ||
            throw(DimensionMismatch(
                "Bond dimension mismatch between sites $n and $(n + 1): " *
                "$right_dim != $left_dim."
            ))
    end

    return MPO{T}(tensors)
end

Base.length(O::MPO) = length(O.tensors)
Base.getindex(O::MPO, inds...) = getindex(O.tensors, inds...)
Base.setindex!(O::MPO, val, inds...) = setindex!(O.tensors, val, inds...)
Base.copy(O::MPO) = MPO(copy.(O.tensors))

function MPO(N::Int, d::Int, χ::Int)
    N > 0 || throw(ArgumentError("Number of sites must be positive."))
    d > 0 || throw(ArgumentError("Physical dimension must be positive."))
    χ > 0 || throw(ArgumentError("Bond dimension must be positive."))

    tensors = [
        Tensor(
            randn(
                ComplexF64,
                n == 1 ? 1 : χ,  # Left virtual bond
                d,               # Output physical index
                d,               # Input physical index
                n == N ? 1 : χ,  # Right virtual bond
            )
        )
        for n in 1:N
    ]

    return MPO(tensors)
end


"""
    apply_MPO(O::MPO, ψ::MPS) -> MPS

Apply the Matrix Product Operator `O` to the Matrix Product State `ψ`.

The MPO tensor at site `n` follows the index convention

    O⁽ⁿ⁾[μₙ₋₁, tₙ, bₙ, μₙ],

with `tₙ` the output physical index and `bₙ` the input physical index.
The MPS tensor follows

    A⁽ⁿ⁾[αₙ₋₁, sₙ, αₙ].

At each site, the MPO input physical index `bₙ` is contracted with the
MPS physical index `sₙ`,

    B⁽ⁿ⁾[μₙ₋₁, tₙ, μₙ, αₙ₋₁, αₙ]
        = ∑ₛ O⁽ⁿ⁾[μₙ₋₁, tₙ, s, μₙ]
             A⁽ⁿ⁾[αₙ₋₁, s, αₙ].

The MPO and MPS virtual indices are then fused,

    (μₙ₋₁, αₙ₋₁) -> βₙ₋₁,
    (μₙ,   αₙ)   -> βₙ,

so that the resulting tensor has the standard MPS form

    B⁽ⁿ⁾[βₙ₋₁, tₙ, βₙ].

Consequently, if the MPO and MPS virtual bond dimensions at a given bond
are `χₙ` and `Dₙ`, respectively, the resulting MPS bond dimension is
`χₙ * Dₙ`.

# Arguments
- `O::MPO`: Matrix Product Operator to apply.
- `ψ::MPS`: Matrix Product State on which the operator acts.

# Returns
- `MPS`: The state `O|ψ⟩`.

# Throws
- `DimensionMismatch`: If `O` and `ψ` have different numbers of sites.
- `DimensionMismatch`: If the MPO input physical dimension and MPS
  physical dimension do not agree at any site.

# Notes
This operation generally increases the MPS bond dimensions. An MPO with
bond dimension `χ` applied to an MPS with bond dimension `D` can produce
bond dimensions up to `χD`.
"""
function apply_MPO(O::MPO{TO}, ψ::MPS{Tψ})::MPS where {TO, Tψ}

    # Both tensor networks must contain the same number of sites.
    length(O) == length(ψ) ||
        throw(DimensionMismatch(
            "MPO and MPS must have the same number of sites."
        ))

    # Number of sites
    N = length(ψ)

    T = promote_type(TO, Tψ)

    tensor_vector = Vector{Tensor{T,3}}(undef, N)

    for n in 1:N
        # Indices O⁽ⁿ⁾[μₙ₋₁, tₙ, bₙ, μₙ] with dimensions (χₙ₋₁, dₙ, dₙ, χₙ)
        O⁽ⁿ⁾ = O[n]
        # Indices A⁽ⁿ⁾[αₙ₋₁, sₙ, αₙ] with dimensions (Dₙ₋₁, dₙ, Dₙ)
        A⁽ⁿ⁾ = ψ[n]

        # The MPO input physical index must match the MPS physical index.
        size(O⁽ⁿ⁾, 3) == size(A⁽ⁿ⁾, 2) ||
            throw(DimensionMismatch(
                "Physical dimension mismatch at site $n."
            ))

        # Local dimensions.
        χₙ₋₁  = size(O⁽ⁿ⁾, 1)
        dₙ   = size(O⁽ⁿ⁾, 2)
        χₙ = size(O⁽ⁿ⁾, 4)

        Dₙ₋₁  = size(A⁽ⁿ⁾, 1)
        Dₙ = size(A⁽ⁿ⁾, 3)

        # Contract along the operator's bottom physical index, bₙ, and the states's physical index, sₙ.
        B⁽ⁿ⁾ = contract(O⁽ⁿ⁾, A⁽ⁿ⁾, 3 => 2)
        # This resulting tensor has indices [μₙ₋₁, tₙ, μₙ, αₙ₋₁, αₙ]. 
        # We want this order: [ μₙ₋₁, αₙ₋₁, tₙ, μₙ, αₙ].
        B⁽ⁿ⁾_perm = permutedims(B⁽ⁿ⁾.data, (1, 4, 2, 3, 5))
        # Now we fuse the left and right virtual indices and leave the physical index in the middle. 
        # (μₙ₋₁, αₙ₋₁) -> βₙ₋₁
        # (μₙ,   αₙ)   -> βₙ
        B⁽ⁿ⁾_fused = reshape(B⁽ⁿ⁾_perm, χₙ₋₁*Dₙ₋₁, dₙ, χₙ*Dₙ)

        tensor_vector[n] = Tensor(B⁽ⁿ⁾_fused)
    end 

    return MPS(tensor_vector)
end
