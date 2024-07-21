using Flux, CUDA, KernelAbstractions, Tullio

struct symbolic_dense
end

function symbolic_kan_kayer(in_dim::Int, out_dim::Int)
    mask = ones(in_dim, out_dim)
end