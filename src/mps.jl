"""
    MPS{T}

Open-boundary Matrix Product State (MPS) with element type `T`.

The MPS consists of one rank-3 tensor for each physical site. The tensor
at site `n` follows the index convention

    (left bond, physical index, right bond)

or, in tensor notation,

    A⁽ⁿ⁾[αₙ₋₁, sₙ, αₙ].

For an MPS with `N` sites, the boundary bond dimensions are fixed to one,

    D₀ = D_N = 1,

so that the tensor dimensions are

    (1, d, D₁),
    (D₁, d, D₂),
    ...
    (Dₙ₋₂, d, Dₙ₋₁),
    (Dₙ₋₁, d, 1),

where `d` is the physical dimension and `Dᵢ` is the dimension of the
bond index `αᵢ`.

# Fields
- `tensors::Vector{Tensor{T,3}}`: Site tensors ordered from left to right
  along the physical chain.
"""
struct MPS{T}
    tensors::Vector{Tensor{T,3}}
end

function MPS(tensors::Vector{Tensor{T, 3}}) where {T}
    # An MPS must contain at least one physical site
    isempty(tensors) && throw(ArgumentError("An MPS must contain at least one tensor."))

    # Open boundary conditions require trivial outer bond dimensions.
    size(first(tensors), 1) == 1 ||
        throw(DimensionMismatch(
            "The left boundary bond dimension must be 1."
        ))

    size(last(tensors), 3) == 1 ||
        throw(DimensionMismatch(
            "The right boundary bond dimension must be 1."
        ))

    # Every site must have the same physical dimension.
    d = size(first(tensors), 2)
    for (n, A) in enumerate(tensors)
        size(A, 2) == d ||
            throw(DimensionMismatch(
                "Physical dimension at site $n does not match the rest of the MPS."
            ))
    end

    # Adjacent virtual bond dimensions must agree.
    for n in 1:length(tensors)-1
        right_dim = size(tensors[n], 3)
        left_dim = size(tensors[n + 1], 1)

        right_dim == left_dim ||
            throw(DimensionMismatch(
                "Bond dimension mismatch between sites $n and $(n + 1): " *
                "$right_dim != $left_dim."
            ))
    end

    return MPS{T}(tensors)
end


Base.length(ψ::MPS) = length(ψ.tensors)
Base.getindex(ψ::MPS, inds...) = getindex(ψ.tensors, inds...)
Base.setindex!(ψ::MPS, val, inds...) = setindex!(ψ.tensors, val, inds...)
Base.copy(ψ::MPS) = MPS(copy.(ψ.tensors))

"""
Physical dimension, d, for an MPS, assumed to be the same for all sites.
Since the convention for the tensor's dimensions is
    A⁽ⁿ⁾[αₙ₋₁, sₙ, αₙ],
we just extract the size along the second axis.
"""
physical_dim(ψ::MPS) = size(first(ψ.tensors), 2)

"""
Bond dimensions of each αₙ index.
"""
bond_dims(ψ::MPS) = [size(A, 3) for A in ψ.tensors[1:end-1]]

"""
All virtual dimensions with the convention that there's a virtual left and right
bond in the edges with dimension 1.
"""
virtual_dims(ψ::MPS) =
    [size(first(ψ.tensors), 1);
     bond_dims(ψ);
     size(last(ψ.tensors), 3)]


function MPS(N::Int, d::Int, D::Int)
    N > 0 || throw(ArgumentError("Number of sites must be positive."))
    d > 0 || throw(ArgumentError("Physical dimension must be positive."))
    D > 0 || throw(ArgumentError("Bond dimension must be positive."))

    tensors = [
        Tensor(
            randn(
                ComplexF64,
                n == 1 ? 1 : D, # Left tensor has left virtual index of a single value
                d,
                n == N ? 1 : D, # Right tensor has right virtual index of a single value
            )
        )
        for n in 1:N
    ]

    return MPS(tensors)
end

function inner(ϕ::MPS, ψ::MPS)
    N = length(ψ)

    N == length(ϕ) || 
        throw(DimensionMismatch(
            "Both MPS's most have the same number of sites."
            ))

    physical_dim(ψ) == physical_dim(ϕ) ||
        throw(DimensionMismatch(
            "Both MPSs must have the same physical dimension."
        ))

    # The first tensor is contracted along the physical dimension (s <-> index 2)
    # as well as the "dummy" dimension (index 1) which makes no sum since its dimension
    # is set to 1.
    # This has two surviving indices, (β₁, α₁)
    E = contract(conj(ϕ[1]), ψ[1], [1 => 1, 2 => 2])

    for n in 2:N
        # In the first step we contract along the right bond, β₁
        # The surviving indices are (αₙ₋₁, sₙ, βₙ₋₁)
        # (βₙ₋₁, αₙ₋₁) × (βₙ₋₁, sₙ, βₙ)
        # -> (αₙ₋₁, sₙ, βₙ)
        E = contract(E, conj(ϕ[n]), 1 => 1)

        # In the second step we contract along the physical s and the α virtual bond
        # The surviving indices are (βₙ, αₙ)
        # (αₙ₋₁, sₙ, βₙ) × (αₙ₋₁, sₙ, αₙ)
        # -> (βₙ, αₙ)
        E = contract(E, ψ[n], [1 => 1, 2 => 2])
    end

    # Open boundary conditions give α_N = β_N = 1.
    return E[1, 1]

end
