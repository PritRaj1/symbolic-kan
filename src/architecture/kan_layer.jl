include("spline.jl")
include("../utils.jl")

using Flux, CUDA, NNlib
using .Spline: extend_grid, coef2curve, curve2coef
using .Utils: device, sparse_mask

struct kan_dense
    in_dim::Int
    out_dim::Int
    num_splines::Int
    degree::Int
    grid
    ε
    coef
    σ_base
    σ_sp
    base_act::Function
    mask
    grid_eps::Float32
    grid_range::Tuple{Float32, Float32}
end

function dense_layer(in_dim::Int, out_dim::Int; num_splines=5, degree=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_act=NNlib.selu, grid_eps=0.02, grid_range=(-1, 1), sparse_init=false)
    grid = range(grid_range[1], grid_range[2], length=num_splines + 1) |> collect |> x -> reshape(x, 1, length(x))
    grid = repeat(grid, in_dim, 1)
    grid = extend_grid(grid, degree)

    ε = (rand(num_splines + 1, in_dim, out_dim) .- 0.5) .* noise_scale ./ num_splines 
    coef = curve2coef(grid[:, degree:end-degree-1] |> permutedims, ε, grid; k=degree)
    
    if sparse_init
        mask = sparse_mask(in_dim, out_dim)
    else
        mask = 1.0
    end

    σ_base = ones(in_dim, out_dim) .* scale_base .* mask
    σ_sp = ones(in_dim, out_dim) .* scale_sp .* mask
    mask = ones(in_dim, out_dim) 

    return kan_dense(in_dim, out_dim, num_splines, degree, grid, ε, coef, σ_base, σ_sp, base_act, mask, grid_eps, grid_range)
end

Flux.@functor kan_dense (grid, coef, σ_base, σ_sp, mask)