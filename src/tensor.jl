"""
    Tensor{T, N}

Dense rank-`N` tensor with elements of type `T`.

# Fields
- `data::Array{T, N}`: Tensor entries.
"""
struct Tensor{T, N}
    data::Array{T, N}
end

"""
    rank(tensor::Tensor) -> Int

Return the rank (number of dimensions) of `tensor`.

# Examples
```jldoctest
julia> t = Tensor(zeros(2, 3, 4));

julia> rank(t)
3
```
"""
function rank(tensor::Tensor{T, N}) where {T, N}
    return N
end

"""
    data_type(tensor::Tensor) -> Type

Return the type of the elements of `tensor`.

# Examples
```jldoctest
julia> t = Tensor([2, 3, 4]);

julia> data_type(t)
Int64
```
"""
function data_type(tensor::Tensor{T, N}) where {T, N}
    return T
end
