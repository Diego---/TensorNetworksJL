"""
    Tensor{T, N}

Dense rank-`N` tensor with elements of type `T`.

# Fields
- `data::Array{T, N}`: Tensor entries.
"""
struct Tensor{T, N}
    data::Array{T, N}
end

Base.ndims(::Tensor{T, N}) where {T, N} = N
Base.eltype(::Type{Tensor{T, N}}) where {T, N} = T

Base.size(t::Tensor) = size(t.data)
Base.size(t::Tensor, dim::Integer) = size(t.data, dim)

Base.getindex(t::Tensor, inds...) = getindex(t.data, inds...)
Base.setindex!(t::Tensor, val, inds...) = setindex!(t.data, val, inds...)