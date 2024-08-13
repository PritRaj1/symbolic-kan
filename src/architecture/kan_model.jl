module KolmogorovArnoldNets

export KAN, KAN_model, prune, update_grid

using CUDA, KernelAbstractions, Lux, LuxCUDA, Random
using Lux, Tullio, NNlib, Random, Statistics, SymPy, Accessors
using Zygote

include("kan_layer.jl")
include("symbolic_layer.jl")
include("../utils.jl")
using .Utils: removeZero, device
using .dense_kan
using .symbolic_layer

struct KAN <: Lux.AbstractExplicitLayer
    widths::Vector{Int}
    depth::Int
    degree::Int
    grid_interval::Int
    base_fcn
    ε_scale::Float32
    μ_scale::Float32
    σ_scale::Float32
    grid_eps::Float32
    grid_range::Tuple{Float32, Float32}
    symbolic_enabled::Bool
    act_fcns
    symbolic_fcns
    bias_trainable::Bool
end

silu = x -> x .* NNlib.sigmoid.(x)

function KAN_model(widths; k=3, grid_interval=5, ε_scale=5f-1, μ_scale=1.0f0, σ_scale=1.0f0, base_act=silu, symbolic_enabled=true, grid_eps=2f-2, grid_range=(-1f0, 1f0), bias_trainable=false)
    depth = length(widths) - 1

    act_fcns = NamedTuple()
    symbolic_fcns = NamedTuple()

    for i in 1:depth
        base_scale = (μ_scale * (1f0 / √(Float32(widths[i]))) 
        .+ σ_scale .* (randn(Float32, widths[i], widths[i + 1]) .* 2f0 .- 1f0) .* (1f0 / √(Float32(widths[i]))))
        
        spline = KAN_Dense(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=1f0, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range)
        @reset act_fcns[Symbol("act_lyr_$i")] = spline
        
        symbolic = SymbolicDense(widths[i], widths[i + 1])
        @reset symbolic_fcns[Symbol("symb_lyr_$i")] = symbolic
    end

    return KAN(widths, depth, k, grid_interval, base_act, ε_scale, μ_scale, σ_scale, grid_eps, grid_range, symbolic_enabled, act_fcns, symbolic_fcns, bias_trainable)
end

function Lux.initialparameters(rng::AbstractRNG, m::KAN)

    # Create one long, flattened named tuple (nested tuples aren't nice with Lux)
    ps = m.bias_trainable ? NamedTuple(Symbol("bias_$i") => zeros(Float32, 1, m.widths[i+1]) for i in 1:m.depth) : NamedTuple()
    
    for i in 1:m.depth
        rng = Random.seed!(rng, i)

        # Parameters for the spline layer
        layer_ps = Lux.initialparameters(rng, m.act_fcns[Symbol("act_lyr_$i")])
        coef, w_base, w_sp = layer_ps[:coef], layer_ps[:w_base], layer_ps[:w_sp]
        @reset ps[Symbol("coef_$i")] = coef
        @reset ps[Symbol("w_base_$i")] = w_base
        @reset ps[Symbol("w_sp_$i")] = w_sp

        # Parameters for the symbolic layer
        symb_ps = Lux.initialparameters(rng, m.symbolic_fcns[Symbol("symb_lyr_$i")])
        @reset ps[Symbol("affine_$i")] = symb_ps

    end

    return ps
end

function Lux.initialstates(rng::AbstractRNG, m::KAN)

    st = NamedTuple(Symbol("mask_$i") => ones(Float32, width) for (i, width) in enumerate(m.widths))
    @reset st[Symbol("acts_1")] = nothing

    if !m.bias_trainable
        for i in 1:m.depth
            @reset st[Symbol("bias_$i")] = zeros(Float32, 1, m.widths[i+1])
        end
    end

    for i in 1:m.depth
        rng = Random.seed!(rng, i)

        @reset st[Symbol("act_scale_$i")] = zeros(Float32, maximum(m.widths[i+1]), maximum(m.widths[i]))
        act_st = Lux.initialstates(rng, m.act_fcns[Symbol("act_lyr_$i")])
        @reset st[Symbol("act_fcn_mask_$i")] = act_st
        symb_st = Lux.initialstates(rng, m.symbolic_fcns[Symbol("symb_lyr_$i")])
        @reset st[Symbol("symb_fcn_mask_$i")] = symb_st

        @reset st[Symbol("acts_$(i+1)")] = nothing
        @reset st[Symbol("pre_acts_$i")] = nothing
        @reset st[Symbol("post_acts_$i")] = nothing
        @reset st[Symbol("post_splines_$i")] = nothing

        @reset st[Symbol("symbolic_acts_$i")] = nothing
    end

    return st
end

function (m::KAN)(x, ps, st)
    """
    Forward pass of the KAN model.

    Args:
        x: A matrix of size (b, in_dim) containing the input data.
        ps: A tuple containing the parameters of the model.
        st: A tuple containing the state of the model.

    Returns:
        y: A matrix of size (b, out_dim) containing the output data.
        scales: A matrix containing the scales for l1 regularization.
        new_st: A tuple containing the new state of the model.
    """
    x, ps, st = device(x), device(ps), device(st)

    max_width = maximum(m.widths)
    scales_init = zeros(Float32, 0, max_width * max_width) |> device

    st = Zygote.ignore() do
        merge(st, Dict(Symbol("acts_1") => copy(x)))
    end

    # Forward pass - use explicit recursion to be more amenable to Zygote, i.e. Φ(Φ(Φ(x)))
    y, scales_out, st = foldl(enumerate(1:m.depth), init=(x, scales_init, st)) do (z, scales, st), (i, _)
        coef = ps[Symbol("coef_$i")]
        w_base = ps[Symbol("w_base_$i")]
        w_sp = ps[Symbol("w_sp_$i")]
        kan_layer = m.act_fcns[Symbol("act_lyr_$i")]

        z_numerical, pre_acts, post_acts_num, post_spline = kan_layer(z, st[Symbol("act_fcn_mask_$i")];  coef=coef, w_base=w_base, w_sp=w_sp)
        
        z_symbolic, symbolic_post_acts = 0f0, 0f0
        if m.symbolic_enabled
            affine = ps[Symbol("affine_$i")]
            symbolic_layer = m.symbolic_fcns[Symbol("symb_lyr_$i")]
            z_symbolic, symbolic_post_acts = symbolic_layer(z, affine, st[Symbol("symb_fcn_mask_$i")])
        end

        # φ(x) + φs(x)
        z_new = z_numerical .+ z_symbolic
        post_acts = post_acts_num .+ symbolic_post_acts

        # Scales for l1 regularisation
        in_range = std(pre_acts, dims=1) .+ 1f-1
        out_range = std(post_acts, dims=1) 
        scale = (out_range ./ in_range)[1, :, :]

        # Pad scales to max width for regularisation
        flat_scale = reshape(scale, :)
        pad = zeros(Float32, (max_width*max_width) - length(flat_scale)) |> device
        flat_scale = vcat(flat_scale, pad)
        new_scales = vcat(scales, reshape(flat_scale, 1, max_width*max_width))
        
        # Add (non-trainable) bias b(x)
        bias = m.bias_trainable ? ps[Symbol("bias_$i")] : st[Symbol("bias_$i")]
        b = repeat(bias, size(z_new, 1), 1)
        z_new = z_new + b

        st_new = Zygote.ignore() do
            merge(st, Dict(
                Symbol("acts_$(i+1)") => copy(z_new),
                Symbol("pre_acts_$i") => copy(pre_acts),
                Symbol("post_acts_$i") => copy(post_acts),
                Symbol("post_splines_$i") => copy(post_spline),
                Symbol("act_scale_$i") => copy(scale)
            ))
        end
        
        return z_new, new_scales, st_new
    end

    return y, scales_out, st
end

function remove_node(st, l, j; verbose=true)
    """
    Remove neuron j from layer l. 
    Masks all incoming and outgoing activation functions for the neuron to zero.

    Args:
        st: State of the model.
        l: Layer index.
        j: Neuron index.

    Returns:
        st: Updated state of the model.
    """
    verbose && println("Removing neuron $(j) from layer $(l)")

    # Remove all incoming connections
    @reset st[Symbol("act_fcn_mask_$(l-1)")][:, j] .= 0.0f0
    @reset st[Symbol("symb_fcn_mask_$(l-1)")][j, :] .= 0.0f0

    # Remove all outgoing connections
    @reset st[Symbol("act_fcn_mask_$(l)")][j, :] .= 0.0f0
    @reset st[Symbol("symb_fcn_mask_$(l)")][:, j] .= 0.0f0

    return st
end

function prune(rng::AbstractRNG, m, ps, st; threshold=3f-2, mode="auto", active_neurons_id=nothing, verbose=true)
    """
    Prune the activation of neuron (l, i, j) based on the threshold.
    If the neuron has a small range of activation, shave off the neuron.

    Args:
        rng: Random number generator.
        m: Model.
        ps: Parameters.
        st: State.
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        threshold: Threshold value.

    Returns:
        model_pruned: Pruned model.
        ps_pruned: Pruned parameters.
        st_pruned: Pruned state.
    """
    st = merge(st, Dict(Symbol("mask_1") => ones(Float32, m.widths[1],)))
    active_neurons_id = []
    push!(active_neurons_id, [1:m.widths[1]...])
    overall_important = nothing

    # Find all neurons with an input and output above the threshold
    for i in 1:m.depth-1
        if mode == "auto"
            scale1 = st[Symbol("act_scale_$i")]
            scale2 = st[Symbol("act_scale_$(i+1)")]
            in_important = ifelse.(maximum(scale1, dims=2)[1, :, :] .> threshold, 1f0,  0f0)
            out_important = ifelse.(maximum(scale2, dims=1)[1, :, :] .> threshold,  1f0,  0f0)
            overall_important = in_important .* out_important
        elseif mode == "manual"
            overall_important = zeros(Float32, m.widths[i+1])
            overall_important[active_neurons_id[i+1]] .= 1f0
        end

        st = merge(st, Dict(Symbol("mask_$(i+1)") => overall_important))
        cart_ind = findall(x -> x > 0f0, overall_important)
        push!(active_neurons_id, [i[1] for i in cart_ind])
    end
    
    push!(active_neurons_id, [1:m.widths[end]...])
    st = merge(st, Dict(Symbol("mask_$(m.depth+1)") => ones(Float32, m.widths[end])))

    if verbose
        println("Active neurons: ", active_neurons_id)
    end

    # Remove neurons with below threshold inputs && outputs
    for i in 1:m.depth
        for j in 1:m.widths[i+1]
            if !(j in active_neurons_id[i+1])
                st = remove_node(st, i+1, j; verbose=verbose)
            end
        end
    end

    # Create pruned models from important subsets
    model_pruned = KAN_model(deepcopy(m.widths); k=m.degree, grid_interval=m.grid_interval, ε_scale=m.ε_scale, μ_scale=m.μ_scale, σ_scale=m.σ_scale, base_act=m.base_fcn, symbolic_enabled=m.symbolic_enabled, grid_eps=m.grid_eps, grid_range=m.grid_range, bias_trainable=m.bias_trainable)
    ps_pruned = Lux.initialparameters(rng, model_pruned)
    st_pruned = Lux.initialstates(rng, model_pruned)

    for i in 1:m.depth
        if i < m.depth
            if m.bias_trainable
                @reset ps_pruned[Symbol("bias_$i")] = ps[Symbol("bias_$i")][:, active_neurons_id[i+1]]
            else
                st_pruned = merge(st, Dict(Symbol("bias_$i") => st[Symbol("bias_$i")][:, active_neurons_id[i+1]]))
            end
        end

        kan_ps = (
            coef = ps[Symbol("coef_$i")],
            w_base = ps[Symbol("w_base_$i")],
            w_sp = ps[Symbol("w_sp_$i")]
        )

        symb_ps = (
            affine = ps[Symbol("affine_$i")]
        )

        new_fcn, ps_new, new_mask = get_subset(model_pruned.act_fcns[i], kan_ps, st_pruned[Symbol("act_fcn_mask_$i")], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.act_fcns[i] = new_fcn

        @reset ps_pruned[Symbol("coef_$i")] = ps_new[:coef]
        @reset ps_pruned[Symbol("w_base_$i")] = ps_new[:w_base]
        @reset ps_pruned[Symbol("w_sp_$i")] = ps_new[:w_sp]

        new_fcn, ps_new, new_symb_mask = get_symb_subset(m.symbolic_fcns[Symbol("symb_lyr_$i")], symb_ps, st_pruned[Symbol("symb_fcn_mask_$i")], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.symbolic_fcns[i] = new_fcn
        @reset ps_pruned[Symbol("affine_$i")] = ps_new
        @reset model_pruned.widths[i] = length(active_neurons_id[i])

        st_pruned = merge(
            st_pruned,
            Dict(
                Symbol("mask_$i") => st[Symbol("mask_$i")],
                Symbol("act_fcn_mask_$i") => new_mask,
                Symbol("symb_fcn_mask_$i") => new_symb_mask
            )
        )
    end

    st_pruned = merge(
        st_pruned,
        Dict(
            Symbol("mask_$(m.depth+1)") => st[Symbol("mask_$(m.depth+1)")]
    ))
    @reset model_pruned.depth = length(model_pruned.widths) - 1

    return model_pruned, ps_pruned, st_pruned
end

function update_grid(model, x, ps, st)
    """
    Update the grid for each b-spline layer in the model.
    """

    for i in 1:model.depth
        _, _, st = model(x, ps, st)
        new_grid, new_coef = update_lyr_grid(model.act_fcns[Symbol("act_lyr_$i")], ps[Symbol("coef_$i")], st[Symbol("acts_$i")])
        @reset ps[Symbol("coef_$i")] = collect(new_coef)
        @reset model.act_fcns[Symbol("act_lyr_$i")].grid = collect(new_grid) |> device
    end
        
    return model, ps
end
    
end