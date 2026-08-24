"""
    Tensor{T,N}

Dense rank-`N` tensor with elements of type `T`.

# Fields
- `data::Array{T,N}`: Tensor entries.
"""
struct Tensor{T,N}
    data::Array{T,N}
end
