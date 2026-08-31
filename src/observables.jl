"""
    two_site_operator_mpo(h, i, N) -> MPO

Construct an `N`-site MPO representing a two-site operator `h` acting
only on neighboring sites `i` and `i+1`.

The two-site operator is split into two MPO tensors using
`split_two_site_operator`. All other sites contain identity tensors.

# Arguments
- `h::AbstractMatrix`: Two-site operator with dimensions `d² × d²`.
- `i::Int`: Left site of the bond on which `h` acts.
- `N::Int`: Total number of sites.

# Returns
- `MPO`: MPO representation of the local operator.
"""
function two_site_operator_mpo(
    h::AbstractMatrix,
    i::Int,
    N::Int,
)::MPO
    1 <= i < N ||
        throw(BoundsError("Bond index $i must satisfy 1 <= i < $N."))

    L, R = split_two_site_operator(h)

    d = size(L, 2)
    T = eltype(L.data)

    tensors = Vector{Tensor{T,4}}(undef, N)

    I_tensor = identity_mpo_tensor(d, T)

    for n in 1:N
        tensors[n] = copy(I_tensor)
    end

    tensors[i]   = L
    tensors[i+1] = R

    return MPO(tensors)
end


"""
    one_site_operator_mpo(O, i, N) -> MPO

Construct an `N`-site MPO representing a one-site operator `O` acting
only on site `i`.

The local `d × d` operator is reshaped into an MPO tensor with trivial
virtual bonds,

    (1, d, d, 1),

while all other sites contain identity tensors.

# Arguments
- `O::AbstractMatrix`: Local one-site operator.
- `i::Int`: Site on which the operator acts.
- `N::Int`: Total number of sites.

# Returns
- `MPO`: MPO representation of the local operator.
"""
function one_site_operator_mpo(
    O::AbstractMatrix,
    i::Int,
    N::Int,
)::MPO
    1 <= i <= N ||
        throw(BoundsError("Site index $i must satisfy 1 <= i <= $N."))

    size(O, 1) == size(O, 2) ||
        throw(DimensionMismatch(
            "The one-site operator must be square."
        ))

    d = size(O, 1)
    T = eltype(O)

    tensors = Vector{Tensor{T,4}}(undef, N)

    I_tensor = identity_mpo_tensor(d, T)

    for n in 1:N
        tensors[n] = copy(I_tensor)
    end

    # Everything except the i'th position are identities
    tensors[i] = Tensor(
        Array(reshape(O, 1, d, d, 1))
    )

    return MPO(tensors)
end


"""
    expectation_nearest_neighbor(ψ, h_terms) -> Number

Compute the expectation value of a nearest-neighbor Hamiltonian

    H = Σᵢ hᵢ,ᵢ₊₁

in the MPS `ψ`.

The convention is

    h_terms[i] = hᵢ,ᵢ₊₁.

Each local Hamiltonian term is represented as an MPO, applied to `ψ`,
and its contribution

    ⟨ψ|hᵢ,ᵢ₊₁|ψ⟩

is evaluated using `inner`.

The returned value is

    ⟨ψ|H|ψ⟩

and assumes `ψ` is normalized if it is to be interpreted as a physical
expectation value.

# Arguments
- `ψ::MPS`: Matrix Product State.
- `h_terms::AbstractVector{<:AbstractMatrix}`: Two-site Hamiltonian
  terms, with `h_terms[i]` acting on sites `i` and `i+1`.

# Returns
- `Number`: The expectation value `⟨ψ|H|ψ⟩`.
"""
function expectation_nearest_neighbor(
    ψ::MPS,
    h_terms::AbstractVector{<:AbstractMatrix},
)::Number
    N = length(ψ)

    length(h_terms) == N - 1 ||
        throw(DimensionMismatch(
            "An N-site nearest-neighbor Hamiltonian must contain N-1 terms."
        ))

    E = zero(inner(ψ, ψ))

    for i in 1:N-1
        H_i = two_site_operator_mpo(h_terms[i], i, N)

        Hψ = apply_MPO(H_i, ψ)

        E += inner(ψ, Hψ)
    end

    return E
end


"""
    expectation_one_body(ψ, O_terms) -> Number

Compute the expectation value of a sum of one-body operators,

    O = Σᵢ Oᵢ,

in the MPS `ψ`.

The convention is

    O_terms[i] = Oᵢ.

Each local operator is embedded into an MPO with identity tensors on all
other sites, applied to `ψ`, and evaluated using `inner`.

# Arguments
- `ψ::MPS`: Matrix Product State.
- `O_terms::AbstractVector{<:AbstractMatrix}`: Local one-site
  operators, one for each physical site.

# Returns
- `Number`: The expectation value `⟨ψ|O|ψ⟩`.
"""
function expectation_one_body(
    ψ::MPS,
    O_terms::AbstractVector{<:AbstractMatrix},
)::Number
    N = length(ψ)

    length(O_terms) == N ||
        throw(DimensionMismatch(
            "An N-site one-body operator sum must contain N terms."
        ))

    expectation = zero(inner(ψ, ψ))

    for i in 1:N
        O_i = one_site_operator_mpo(O_terms[i], i, N)
        Oψ = apply_MPO(O_i, ψ)

        expectation += inner(ψ, Oψ)
    end

    return expectation
end
