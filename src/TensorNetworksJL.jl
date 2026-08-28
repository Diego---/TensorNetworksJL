module TensorNetworksJL

include("tensor.jl")
include("contractions.jl")
include("mps.jl")

export Tensor, contract, MPS, inner, svdcompress, entanglement_entropy

end
