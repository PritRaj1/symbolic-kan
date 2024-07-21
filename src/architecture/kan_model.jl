using FLux, CUDA, KernelAbstractions, Tullio, NNlib, Random
using Flux: Chain, Dense

include("kan_layer.jl")

using .kan_dense: b_spline_layer, update_grid!

struct KAN
end

function KAN(widths; k=3, grid_interval=3, ε_scale=0.1, μ_scale=0.0, σ_scale=1.0, base_act=NNlib.selu, symbolic=true, grid_eps=1.0, grid_range=(-1, 1), sparse_init=false, init_seed=0)
    Random.seed!(init_seed)

    biases = []
    act_fcns = []
    depth = length(widths) - 1 

    for i in 1:depth
        base_scale = (μ_scale * (1 / √(widths[i])) 
        + σ_scale * (randn(width[i], width[i + 1]) * 2 .- 1) * (1 / √(width[i])))
        spline = b_spline_layer(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=base_scale, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range, sparse_init=sparse_init)
        push!(act_fcns, spline)
        bias = Chain(
            permutedims,
            Dense(widths[i + 1] => widths[i + 1], identity; bias=false, init=zeros32),
            permutedims,
        )
        push!(biases, bias)
    end



    
    


end