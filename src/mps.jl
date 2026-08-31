using LinearAlgebra

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

function inner(ϕ::MPS, ψ::MPS)::Number
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

"""
    truncated_svd(A::AbstractMatrix, ϵ::Real, Dmax::Union{Nothing, Int})

Computes the Singular Value Decomposition (SVD) of a matrix `A` and truncates small singular values.

Singular values ≤ `ϵ` are discarded, except that at least the largest singular value is always retained 
to prevent zeroing out the entire state.

An optional maximum bond size can be assigned, and at most `Dmax` virtual bonds will be used.

# Arguments
- `A::AbstractMatrix`: The matrix to be decomposed.
- `ϵ::Real`: The truncation tolerance.
- `Dmax::Union{Nothing, Int}`: Optional maximum bond size.

# Returns
- `U::Matrix`: The truncated left singular vectors.
- `S::Vector`: The truncated singular values.
- `Vt::Matrix`: The truncated right singular vectors (adjoint of V).

Does not mutate any arguments.
"""
function truncated_svd(
    A::AbstractMatrix,
    ϵ::Real,
    Dmax::Union{Nothing, Int}=nothing
    )
    ϵ >= 0 || throw(ArgumentError("SVD tolerance must be nonnegative."))

    if Dmax !== nothing
        Dmax > 0 ||
            throw(ArgumentError(
                "Maximum bond dimension must be positive."
            ))
    end

    F = svd(A)

    # We truncate the singular values that are less than the tolerance
    # If all are below it, we keep only the biggest one
    χ = max(1, count(>(ϵ), F.S))

    # Impose the maximum bond dimension, if requested.
    if Dmax !== nothing
        χ = min(χ, Dmax)
    end

    # We truncate S, the columns of U, and the rows of Vt
    U  = F.U[:,1:χ]
    S  = F.S[1:χ]
    Vt = F.Vt[1:χ,:]

    return U, S, Vt
end

"""
    left_svd_step!(ψ::MPS, n::Int, ϵ::Real, Dmax::Union{Nothing, Int})

Performs a left-to-right SVD sweep step at site `n`, pushing the orthogonality center to site `n+1`.

The tensor at site `n` with indices `(αₙ₋₁, sₙ, αₙ)` is reshaped into a matrix with fused indices 
`((αₙ₋₁, sₙ), αₙ)` and decomposed via SVD. Singular values ≤ `ϵ` are discarded, except that at least 
the largest singular value is retained. The left unitary matrix replaces the tensor at site 
`n` (leaving it left-canonical). The singular values and right unitary matrix are absorbed into the 
tensor at site `n+1`.

# Arguments
- `ψ::MPS`: The Matrix Product State.
- `n::Int`: The index of the site to decompose.
- `ϵ::Real`: The truncation tolerance.
. `Dmax::Union{Nothing, Int}`: Optional maximum bond size.

# Returns
- The updated tensor at site `n+1` (as a result of the assignment).

Mutates the MPS `ψ` in place.
"""
function left_svd_step!(
    ψ::MPS,
    n::Int,
    ϵ::Real,
    Dmax::Union{Nothing, Int}=nothing
    )
    # First we reshape the current tensor
    A = ψ[n]
    D_left, d, D_right = size(A)

    # A_mat has the left and center indices fused, 
    # so now it has indices ((αₙ₋₁, sₙ), αₙ)
    A_mat = reshape(A.data, D_left * d, D_right)

    # We perform an SVD and truncation
    U, S, Vt = truncated_svd(A_mat, ϵ, Dmax)
    χ = length(S)

    # U_tensor has indices (αₙ₋₁, sₙ, γₙ), we must reshape
    ψ[n] = Tensor(reshape(U, D_left, d, χ))

    # SV† has indices (γₙ, αₙ)
    SV = Tensor(Diagonal(S) * Vt)

    # We contract along the α index
    ψ[n+1] = contract(SV, ψ[n+1], 2 => 1)
end

"""
    right_svd_step!(ψ::MPS, n::Int, ϵ::Real, Dmax::Union{Nothing, Int})

Performs a right-to-left SVD sweep step at site `n`, pushing the orthogonality center to site `n-1`.

The tensor at site `n` with indices `(αₙ₋₁, sₙ, αₙ)` is reshaped into a matrix with fused indices 
`(αₙ₋₁, (sₙ, αₙ))` and decomposed via SVD. Singular values ≤ `ϵ` are discarded, except that at least 
the largest singular value is retained. The right unitary matrix replaces the tensor at site 
`n` (leaving it right-canonical). The left unitary matrix and singular values are absorbed into the 
tensor at site `n-1`.

# Arguments
- `ψ::MPS`: The Matrix Product State.
- `n::Int`: The index of the site to decompose.
- `ϵ::Real`: The truncation tolerance.
- `Dmax::Union{Nothing, Int}`: Optional maximum bond size.

# Returns
- The updated tensor at site `n-1` (as a result of the assignment).

Mutates the MPS `ψ` in place.
"""
function right_svd_step!(
    ψ::MPS,
    n::Int,
    ϵ::Real,
    Dmax::Union{Nothing, Int}=nothing
    )
    # First we reshape the current tensor
    A = ψ[n]
    D_left, d, D_right = size(A)

    # A_mat has the right and center indices fused, 
    # so now it has indices (αₙ₋₁, (sₙ, αₙ))
    A_mat = reshape(A.data, D_left, d * D_right)

    # We perform an SVD and truncation
    U, S, Vt = truncated_svd(A_mat, ϵ, Dmax)
    χ = length(S)

    # Vt_tensor has indices (γₙ₋₁, sₙ, αₙ), we must reshape
    ψ[n] = Tensor(reshape(Vt, χ, d, D_right))

    # US has indices (αₙ₋₁, γₙ₋₁)
    US = Tensor(U * Diagonal(S))

     # We contract along the α index
    ψ[n-1] = contract(ψ[n-1], US, 3 => 1)
end


"""
    svdcompress(ψ::MPS, cut::Int, ϵ::Real=0, Dmax::Union{Nothing, Int})::MPS

Compresses an MPS using SVD truncation, targeting an orthogonality center at a specified `cut`.

The function performs a sequence of SVD steps:
1. Left sweeps from site `1` to `cut-1` bring the left portion of the MPS toward left-canonical form.
2. Right sweeps from site `N` down to `cut+1` bring the right portion of the MPS toward right-canonical form.
During these sweeps, singular values ≤ `ϵ` are discarded, except that at least the largest singular value is 
always retained. 

# Arguments
- `ψ::MPS`: The original Matrix Product State.
- `cut::Int`: The site index forming the boundary of the left and right sweeps.
- `ϵ::Real`: The threshold below which singular values are truncated (default is 0).
- `Dmax::Union{Nothing, Int}`: Optional maximum bond size.

# Returns
- A new, compressed `MPS` instance.

Does not mutate the original MPS (operates on a copy).
"""
function svdcompress(
    ψ::MPS,
    cut::Int,
    ϵ::Real=0,
    Dmax::Union{Nothing, Int}=nothing
    )::MPS
    N = length(ψ)

    1 <= cut <= N ||
        throw(BoundsError(
            "Cut must satisfy 1 <= cut <= $N."
        ))

    ψ_copy = copy(ψ)

    for n in 1:cut-1
        left_svd_step!(ψ_copy, n, ϵ, Dmax)
    end

    for n in N:-1:cut+1
        right_svd_step!(ψ_copy, n, ϵ, Dmax)
    end

    return ψ_copy
end

"""
    entanglement_entropy(ψ::MPS, cut::Int, ϵ::Real, Dmax::Union{Nothing, Int})::Real

Calculates the entanglement entropy between subsystems ψ[1:`cut`] and ψ[`cut`+1:N].

# Arguments
- `ψ::MPS`: The original Matrix Product State.
- `cut::Int`: The site index forming the boundary of the left and right sweeps.
- `ϵ::Real`: The threshold below which singular values are truncated (default is 0).
- `Dmax::Union{Nothing, Int}`: Optional maximum bond size.

# Returns
- The entanglement entropy of the compressed `MPS` instance between subsystems separated
at site `cut`.
"""
function entanglement_entropy(
    ψ::MPS,
    cut::Int,
    ϵ::Real=0,
    Dmax::Union{Nothing, Int}=nothing
    )::Real
    # First, bring the MPS into mixed-canonical form at the cut
    ψ_mixed = svdcompress(ψ, cut, ϵ, Dmax)
    
    # Extract the center tensor
    A_center = ψ_mixed[cut]
    D_left, d, D_right = size(A_center)
    
    # Reshape to matrix by fusing the left and physical bond ((αₙ₋₁, sₙ), αₙ)
    A_mat = reshape(A_center.data, D_left * d, D_right)
    
    # Perform the SVD to get the Schmidt coefficients (singular values)
    F = svd(A_mat)
    λ = F.S  # These are the Schmidt coefficients
    
    # Calculate probabilities p_i = λ_i^2
    p = λ .^ 2
    
    # Normalize just in case the original MPS wasn't perfectly normalized
    p = p ./ sum(p)
    
    # Calculate Von Neumann entropy: -sum(p * log(p))
    # We only sum over p > 0 to avoid log(0) errors
    S_vN = -sum(x * log(x) for x in p if x > 0)
    
    return S_vN
end


"""
    normalize_mps(ψ::MPS) -> MPS

Return a normalized copy of the MPS `ψ`.

The norm is computed as

    ||ψ|| = sqrt(⟨ψ|ψ⟩),

and one site tensor is divided by this value. Scaling a single MPS
tensor scales the complete many-body state by the same factor.

# Arguments
- `ψ::MPS`: Matrix Product State to normalize.

# Returns
- `MPS`: A copy representing `ψ / ||ψ||`.

# Throws
- `ArgumentError`: If the state has zero norm.
"""
function normalize_mps(ψ::MPS)::MPS
    ψ_norm² = real(inner(ψ, ψ))

    ψ_norm² > 0 ||
        throw(ArgumentError(
            "Cannot normalize an MPS with zero norm."
        ))

    ψ_norm = sqrt(ψ_norm²)

    ψ_normalized = copy(ψ)

    ψ_normalized[1] =
        Tensor(ψ_normalized[1].data / ψ_norm)

    return ψ_normalized
end
