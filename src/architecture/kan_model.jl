module KolmogorovArnoldNets

export KAN, fwd!, update_grid!

using Flux, CUDA, KernelAbstractions, Tullio, NNlib, Random, Statistics
using FunctionWrappers: FunctionWrapper

include("kan_layer.jl")
include("symbolic_layer.jl")
using .dense_kan: b_spline_layer, update_lyr_grid!
using .symbolic_layer: symbolic_kan_layer

mutable struct KAN_
    widths::Vector{Int}
    depth::Int
    grid_interval::Int
    base_fcn::FunctionWrapper{Float64, Tuple{Float64}}
    act_fcns::Vector{Any}
    biases::Vector{Any}
    symbolic_fcns::Vector{Any}
    symbolic_enabled::Bool
    acts::Vector{AbstractArray}
    pre_acts::Vector{AbstractArray}
    post_acts::Vector{AbstractArray}
    post_splines::Vector{AbstractArray}
    acts_scale::Vector{AbstractArray}
end

function KAN(widths; k=3, grid_interval=3, ε_scale=0.1, μ_scale=0.0, σ_scale=1.0, base_act=NNlib.selu, symbolic_enabled=true, grid_eps=1.0, grid_range=(-1, 1), sparse_init=false, init_seed=nothing)

    biases = []
    act_fcns = []
    depth = length(widths) - 1 

    for i in 1:depth
        isnothing(init_seed) ? Random.seed!() : Random.seed!(init_seed + i)
        base_scale = (μ_scale * (1 / √(widths[i])) 
        .+ σ_scale .* (randn(widths[i], widths[i + 1]) .* 2 .- 1) .* (1 / √(widths[i])))
        spline = b_spline_layer(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=base_scale, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range, sparse_init=sparse_init)
        push!(act_fcns, spline)
        bias = zeros(widths[i + 1], 1)
        push!(biases, bias)
    end

    symbolic = []
    for i in 1:depth
        push!(symbolic, symbolic_kan_layer(widths[i], widths[i + 1]))
    end

    return KAN_(widths, depth, grid_interval, base_act, act_fcns, biases, symbolic, symbolic_enabled, [], [], [], [], [])
end

Flux.@functor KAN_

function fwd!(model::KAN_, x)
    model.acts = [x]
    model.pre_acts = []
    model.post_acts = []
    model.post_splines = []
    model.acts_scale = []
    x_eval = copy(x)

    for i in 1:model.depth
        # Evaluate b_spline at x
        x_numerical, pre_acts, post_acts_numerical, postspline = model.act_fcns[i](model.acts[i])

        # Evaluate symbolic layer at x
        if model.symbolic_enabled
            x_symbolic, post_acts_symbolic = model.symbolic_fcns[i](model.acts[i])
        else
            x_symbolic, post_acts_symbolic = 0.0, 0.0
        end

        x_eval = x_numerical .+ x_symbolic
        post_acts = post_acts_numerical .+ post_acts_symbolic

        in_range = std(pre_acts, dims=1)[1, :, :] .+ 0.1
        out_range = std(post_acts, dims=1)[1, :, :] .+ 0.1
        push!(model.acts_scale, (out_range ./ in_range)) 
        push!(model.pre_acts, pre_acts)
        push!(model.post_acts, post_acts)
        push!(model.post_splines, postspline)

        # Add bias
        b = model.biases[i]
        @tullio x_eval[n, m] += b[m, 1]
        push!(model.acts, x_eval)
    end

    return x
end

function update_grid!(model::KAN_, x)
    """
    Update the grid for each b-spline layer in the model.
    """
 
    for i in 1:model.depth
        _ = fwd!(model, x)
        update_lyr_grid!(model.act_fcns[i], model.acts[i])
    end

end

end