"""
    contract(T::Tensor, M::Tensor, indices::Vector{Pair{Int,Int}})
    contract(T::Tensor, M::Tensor, index::Pair{Int,Int})

Contract selected pairs of indices of tensors `T` and `M`.

Each pair `i => j` specifies that index `i` of `T` is contracted with
index `j` of `M`. The corresponding dimensions must be equal.

# Arguments

- `T::Tensor`: First tensor.
- `M::Tensor`: Second tensor.
- `indices::Vector{Pair{Int,Int}}`: Pairs specifying the indices to
  contract. For example, `2 => 1` contracts the second index of `T`
  with the first index of `M`.
- `index::Pair{Int,Int}`: Convenience form for contracting a single
  pair of indices.

# Returns

- `Tensor`: The tensor obtained after contracting the selected indices.

# Examples

Contract the last index of a rank-3 tensor with the first index of a
rank-2 tensor:

```julia
T = Tensor(rand(2, 3, 4))
M = Tensor(rand(4, 5))

C = contract(T, M, 3 => 1)

size(C) == (2, 3, 5)```
"""
function contract(T::Tensor, M::Tensor, indices::Vector{Pair{Int,Int}})
    # Identify contracted and free indices
    T_contracted_idx = first.(indices)
    M_contracted_idx = last.(indices)

    T_free_idx = setdiff(1:ndims(T), T_contracted_idx)
    M_free_idx = setdiff(1:ndims(M), M_contracted_idx)

    # Validate the contraction
    for (T_idx, M_idx) in indices
        1 <= T_idx <= ndims(T) || throw(BoundsError(T, T_idx))
        1 <= M_idx <= ndims(M) || throw(BoundsError(M, M_idx))

        size(T, T_idx) == size(M, M_idx) || throw(DimensionMismatch("""
        Cannot contract index $T_idx of T with index $M_idx of M:
        - Dimension of T (leg $T_idx): $(size(T, T_idx))
        - Dimension of M (leg $M_idx): $(size(M, M_idx))
        Leg dimensions must match.
        """))
    end

    if length(unique(T_contracted_idx)) != length(T_contracted_idx) || 
        length(unique(M_contracted_idx)) != length(M_contracted_idx)
        throw(ArgumentError(
            "Repeated index detected in contraction pairs. " *
            "Each tensor leg can only be contracted once."
        ))
    end

    # Permute indices
    # T -> (free indices, contracted indices)
    # M -> (contracted indices, free indices)
    T_perm = permutedims(
        T.data,
        (T_free_idx..., T_contracted_idx...),
    )

    M_perm = permutedims(
        M.data,
        (M_contracted_idx..., M_free_idx...),
    )

    # Fuse indices so that the contraction becomes matrix multiplication
    T_row_dim = prod(
        size(T, i) for i in T_free_idx;
        init=1,
    )

    T_column_dim = prod(
        size(T, i) for i in T_contracted_idx;
        init=1,
    )

    M_row_dim = prod(
        (size(M, i) for i in M_contracted_idx);
        init=1,
    )

    M_column_dim = prod(
        (size(M, i) for i in M_free_idx);
        init=1,
    )

    T_fused = reshape(T_perm, T_row_dim, T_column_dim)
    M_fused = reshape(M_perm, M_row_dim, M_column_dim)

    # Contract, which is now just a matrix multiplication
    C_fused = T_fused * M_fused

    # Unfuse the remaining indices
    T_free_dims = [size(T, i) for i in T_free_idx]
    M_free_dims = [size(M, i) for i in M_free_idx]

    C_dims = (T_free_dims..., M_free_dims...)

    C = Tensor(reshape(C_fused, C_dims))

    return C
end


contract(T::Tensor, M::Tensor, index::Pair{Int,Int}) = contract(T, M, [index])
