module KolmogorovArnoldNets

export KAN, fwd!, update_grid!, fix_symbolic!, prune

using Flux, Tullio, NNlib, Random, Statistics
# using CUDA, KernelAbstractions

include("kan_layer.jl")
include("symbolic_layer.jl")
using .dense_kan: b_spline_layer, update_lyr_grid!, get_subset, fwd
using .symbolic_layer: symbolic_kan_layer, lock_symbolic!, symb_fwd

mutable struct KAN_
    widths::Vector{Int}
    depth::Int
    grid_interval::Int
    base_fcn
    act_fcns
    biases::Vector{AbstractArray{Float32}}
    symbolic_fcns
    symbolic_enabled::Bool
    acts::Vector{AbstractArray{Float32}}
    pre_acts::Vector{AbstractArray{Float32}}
    post_acts::Vector{AbstractArray{Float32}}
    post_splines::Vector{AbstractArray{Float32}}
    act_scale::AbstractArray{Float32}
    mask::AbstractArray{Float32}
end

function KAN(widths; k=3, grid_interval=3, ε_scale=0.1, μ_scale=0.0, σ_scale=1.0, base_act=NNlib.selu, symbolic_enabled=true, grid_eps=1.0, grid_range=(-1, 1), sparse_init=false, init_seed=nothing)

    biases = []
    act_fcns = []
    symbolic = []
    depth = length(widths) - 1 

    for i in 1:depth
        isnothing(init_seed) ? Random.seed!() : Random.seed!(init_seed + i)
        base_scale = (μ_scale * (1 / √(widths[i])) 
        .+ σ_scale .* (randn(widths[i], widths[i + 1]) .* 2 .- 1) .* (1 / √(widths[i])))
        spline = b_spline_layer(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=base_scale, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range, sparse_init=sparse_init)
        push!(act_fcns, spline)
        bias = zeros(1, widths[i + 1])
        push!(biases, bias)
        push!(symbolic, symbolic_kan_layer(widths[i], widths[i + 1]))
    end

    return KAN_(widths, depth, grid_interval, base_act, act_fcns, biases, symbolic, symbolic_enabled, [], [], [], [], zeros(Float32, 0, 0), ones(widths[end], )
end

Flux.@functor KAN_ (biases, act_fcns)

using Zygote: @nograd

@nograd function add_to_array!(arr, x)
    return push!(arr, x)
end

function PadToShape(arr, shape)
    """
    Padding for act scales. 
    Pad zeros, because we only care about max scales in regularisation.
    """
    pad = shape .- size(arr)

    # Pad zeros to the first dimension
    zeros_1 = zeros(Float32, 0, pad[2], size(arr, 3))
    array = cat(arr, zeros_1, dims=(1, 2))

    # Pad zeros to the second dimension
    zeros_2 = zeros(Float32, 0, size(array, 2), pad[3])
    return cat(array, zeros_2, dims=(1, 3))
end

function fwd!(model, x)
    model.pre_acts = []
    model.post_acts = []
    model.post_splines = []
    model.act_scale = zeros(Float32, 0, maximum(model.widths), maximum(model.widths))
    model.acts = [x]
    x_eval = copy(x)

    for i in 1:model.depth
        # spline(x)
        x_numerical, pre_acts, post_acts_numerical, postspline = fwd(model.act_fcns[i], x_eval)

        # Evaluate symbolic layer at x
        x_symbolic, post_acts_symbolic = 0.0, 0.0
        if model.symbolic_enabled
            x_symbolic, post_acts_symbolic = symb_fwd(model.symbolic_fcns[i], x_eval)
        end

        x_eval = x_numerical .+ x_symbolic
        post_acts = post_acts_numerical .+ post_acts_symbolic

        # Scales for l1 regularisation
        in_range = std(pre_acts, dims=1).+ 0.1
        out_range = std(post_acts, dims=1) .+ 0.1
        scales = PadToShape(out_range ./ in_range, (1, maximum(model.widths), maximum(model.widths)))
        model.act_scale = vcat(model.act_scale, scales)
        
        add_to_array!(model.pre_acts, pre_acts)
        add_to_array!(model.post_acts, post_acts)
        add_to_array!(model.post_splines, postspline)

        # Add bias b(x)
        b = repeat(model.biases[i], size(x_eval, 1), 1)
        x_eval = @tullio res[m, n] := x_eval[m, n] + b[m, n]

        add_to_array!(model.acts, copy(x_eval))
    end

    return x_eval
end

function update_grid!(model, x)
    """
    Update the grid for each b-spline layer in the model.
    """
 
    for i in 1:model.depth
        _ = fwd!(model, x)
        update_lyr_grid!(model.act_fcns[i], model.acts[i])
    end
end

function set_mode!(model, l, i, j, mode; mask_n=nothing)
    """
    Set neuron (l, i, j) to mode.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        mode: 'n' (numeric) or 's' (symbolic) or 'ns' (combined)
        mask_n: Magnitude of the mask for numeric neurons.
    """

    if mode == "s"
        mask_n = 0.0
        mask_s = 1.0
    elseif mode == "n"
        mask_n = 1.0
        mask_s = 0.0
    elseif mode == "sn" || mode == "ns"
        if isnothing(mask_n)
            mask_n = 1.0
        else
            mask_n = mask_n
        end
        mask_s = 1.0
    else
        mask_n = 0.0
        mask_s = 0.0
    end

    model.act_fcns[l].mask[i, j] = mask_n
    model.symbolic_fcns[l].mask[j, i] = mask_s
end

function fix_symbolic!(model, l, i, j, fcn_name; fit_params=true, α_range=(-10, 10), β_range=(-10, 10), grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=true)
    """
    Set the activation for element (l, i, j) to a fixed symbolic function.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        fcn_name: Name of the symbolic function.
        fit_params: Fit the parameters of the symbolic function.
        α_range: Range of the α parameter in fit.
        β_range: Range of the β parameter in fit.
        grid_number: Number of grid points in fit.
        iterations: Number of iterations in fit.
        μ: Step size in fit.
        random: Random setting.
        verbose: Print updates.

    Returns:
        R2 (or nothing): Coefficient of determination.
    """
    set_mode!(model, l, i, j, "s")
    
    if !fit_params
        lock_symbolic!(model.symbolic_fcns[l], i, j, fcn_name)
        return nothing
    else
        x = model.acts[l]
        y = model.post_acts[l][:, i, j]
        R2 = lock_symbolic!(model.symbolic_fcns[l], i, j, fcn_name; x=x, y=y, α_range=α_range, β_range=β_range, μ=μ, random=random, seed=seed, verbose=verbose)
        return R2
    end 
end

function unfix_symbolic!(model, l, i, j)
    """
    Unfix the symbolic function for element (l, i, j).

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
    """
    set_mode!(model, l, i, j, "n")
end

function unfix_symb_all!(model)
    """
    Unfix all symbolic functions in the model.
    """
    for l in 1:model.depth
        for i in 1:model.widths[l]
            for j in 1:model.widths[l + 1]
                unfix_symbolic!(model, l, i, j)
            end
        end
    end
end

function remove_node!(model, l, j)
    """
    Remove neuron j from layer l. 
    Masks all incoming and outgoing activation functions for the neuron to zero.

    Args:
        l: Layer index.
        j: Neuron index.
    """
    # Remove all outgoing connections
    for i in 1:model.widths[l]
        set_mode!(model, l, i, j, "remove")
    end

    for o in 1:model.widths[l - 1]
        model.act_fcns[l-1].mask[j, o] = 0.0  
        model.symbolic_fcns[l-1].mask[o, j] = 0.0
    end
end

function prune(model; threshold=1e-2, mode="auto", active_neurons_id=nothing)
    """
    Prune the activation of neuron (l, i, j) based on the threshold.
    If the neuron has a small range of activation, shave off the neuron.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        threshold: Threshold value.

    Returns:
        model_pruned: Pruned model.
    """
    mask = []
    add_to_array!(mask, ones(model.widths[1], ))
    active_neurons_id = [[1:model.widths[1]...]]

    for i in 1:model.depth-1
        if mode == "auto"
            in_important = ifelse.(maximum(model.act_scale[i, :, :], dims=2)[1, :, :] .> threshold, 1.0, 0.0)
            out_important = ifelse.(maximum(model.act_scale[i+1, :, :], dims=1)[1, :, :] .> threshold, 1.0, 0.0)
            overall_important = in_important .* out_important
        elseif mode == "manual"
            overall_important = zeros(Bool, model.widths[i+1])
            overall_important[active_neurons_id[i+1]] .= true
        end

        push!(mask, overall_important)
        cart_ind = findall(x -> x == true, overall_important)
        push!(active_neurons_id, [i[1] for i in cart_ind])
    end

    push!(mask, ones(model.widths[end], ))
    model.mask = mask

    for i in 1:model.depth-1
        for j in 1:model.widths[i+1]
            if !active_neurons_id[i+1][j] in active_neurons_id[i+1]
                remove_node!(model, i+1, j)
            end
        end
    end

    model_pruned = deepcopy(model)

    for i in 1:size(model.act_scale, 1)
        if i < size(model.act_scale, 1)
            model_pruned.biases[i] = model.biases[i][:, active_neurons_id[i+1]]
        end

        model_pruned.act_fcns[i] = get_subset(model.act_fcns[i], active_neurons_id[i], active_neurons_id[i+1])
        model_pruned.widths[i] = length(active_neurons_id[i])
        model_pruned.symbolic_fcns[i] = get_symb_subset(model.symbolic_fcns[i], active_neurons_id[i+1], active_neurons_id[i])
    end

    return model_pruned
end
end