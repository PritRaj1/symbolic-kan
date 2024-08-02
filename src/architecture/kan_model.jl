module KolmogorovArnoldNets

export KAN, KAN_model, prune, update_grid

using Lux, Tullio, NNlib, Random, Statistics, SymPy, Accessors
using Zygote: @nograd
# using CUDA, KernelAbstractions

include("kan_layer.jl")
include("symbolic_layer.jl")
include("../utils.jl")
using .Utils: removeNaN, removeZero
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
end

function KAN_model(widths; k=3, grid_interval=3, ε_scale=0.1f0, μ_scale=0.0f0, σ_scale=1.0f0, base_act=NNlib.hardtanh, symbolic_enabled=true, grid_eps=1.0f0, grid_range=(-1f0, 1f0))
    depth = length(widths) - 1

    act_fcns = []
    symbolic_fcns = []

    for i in 1:depth
        base_scale = (μ_scale * (1 / √(widths[i])) 
        .+ σ_scale .* (randn(Float32, widths[i], widths[i + 1]) .* 2 .- 1) .* (1 / √(widths[i])))
        
        spline = KAN_Dense(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=base_scale, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range)
        push!(act_fcns, spline)
        
        symbolic = SymbolicDense(widths[i], widths[i + 1])
        push!(symbolic_fcns, symbolic)
    end

    return KAN(widths, depth, k, grid_interval, base_act, ε_scale, μ_scale, σ_scale, grid_eps, grid_range, symbolic_enabled, act_fcns, symbolic_fcns)
end

function Lux.initialparameters(rng::AbstractRNG, m::KAN)
    act_fcns_ps = NamedTuple(Symbol("layer_$i") => Lux.initialparameters(rng, m.act_fcns[i]) for i in 1:m.depth)
    symbolic_fcns_ps =  NamedTuple(Symbol("layer_$i") => Lux.initialparameters(rng, m.symbolic_fcns[i]) for i in 1:m.depth)
    biases = NamedTuple(Symbol("layer_$i") => zeros(Float32, 1, m.widths[i+1]) for i in 1:m.depth)

    ps = (
        act_fcns_ps=act_fcns_ps,
        symbolic_fcns_ps=symbolic_fcns_ps,
        biases=biases
    )
    return ps
end

function Lux.initialstates(rng::AbstractRNG, m::KAN)
    acts_fcns_st = [Lux.initialstates(rng, m.act_fcns[i]) for i in 1:m.depth]
    symbolic_fcns_st = [Lux.initialstates(rng, m.symbolic_fcns[i]) for i in 1:m.depth]

    st = (
        act_fcns_st=acts_fcns_st,
        symbolic_fcns_st=symbolic_fcns_st,
        acts = [],
        pre_acts = [],
        post_acts = [],
        post_splines = [],
        act_scale = zeros(Float32, 0, maximum(m.widths), maximum(m.widths)),
        mask = [ones(Float32, widths) for widths in m.widths[1:end]],
    )

    return st
end

@nograd function add_to_array!(arr, x)
    return push!(arr, x)
end

function PadToShape(arr, shape)
    pad = shape .- size(arr)
    zeros_1 = zeros(Float32, 0, pad[2], size(arr, 3))
    array = cat(arr, zeros_1, dims=(1, 2))
    zeros_2 = zeros(Float32, 0, size(array, 2), pad[3])
    return cat(array, zeros_2, dims=(1, 3))
end

function (m::KAN)(x, ps, st)
    x_eval = copy(x)
    acts_arr = [x,]
    pre_acts_arr = []
    post_acts_arr = []
    post_splines_arr = []
    act_scale_arr = zeros(Float32, 0, maximum(m.widths), maximum(m.widths))

    for i in 1:m.depth
        # spline(x)
        x_numerical, spline_st = m.act_fcns[i](x_eval, ps.act_fcns_ps[Symbol("layer_$i")], st.act_fcns_st[i])
        
        # Evaluate symbolic layer at x
        x_symbolic, symbolic_st = Float32(0.0), Float32(0.0)
        if m.symbolic_enabled
            x_symbolic, symbolic_st = m.symbolic_fcns[i](x_eval, ps.symbolic_fcns_ps[Symbol("layer_$i")], st.symbolic_fcns_st[i])
        end

        x_eval = x_numerical .+ x_symbolic
        post_acts = spline_st.post_acts .+ symbolic_st.post_acts

        # Scales for l1 regularisation
        in_range = std(spline_st.pre_acts, dims=1)
        out_range = std(post_acts, dims=1)
        in_range = removeZero.(in_range; ε=1e-1)
        scales = PadToShape(out_range ./ in_range, (1, maximum(m.widths), maximum(m.widths)))
        act_scale_arr = vcat(act_scale_arr, scales)
        
        add_to_array!(pre_acts_arr, spline_st.pre_acts)
        add_to_array!(post_acts_arr, post_acts)
        add_to_array!(post_splines_arr, spline_st.post_spline)

        # Add bias b(x)
        b = repeat(ps.biases[Symbol("layer_$i")], size(x_eval, 1), 1)
        x_eval = @tullio res[m, n] := x_eval[m, n] + b[m, n]

        add_to_array!(acts_arr, copy(x_eval))
    end

    new_st = (
        act_fcns_st=st.act_fcns_st,
        symbolic_fcns_st=st.symbolic_fcns_st,
        acts = acts_arr,
        pre_acts = pre_acts_arr,
        post_acts = post_acts_arr,
        post_splines = post_splines_arr,
        act_scale = act_scale_arr,
        mask = st.mask,
    )

    return x_eval, new_st
end

function remove_node(st, l, j; verbose=true)
    """
    Remove neuron j from layer l. 
    Masks all incoming and outgoing activation functions for the neuron to zero.

    Args:
        l: Layer index.
        j: Neuron index.
    """
    verbose && println("Removing neuron $(j) from layer $(l)")

    # Remove all incoming connections
    @reset st.act_fcns_st[l-1].mask[:, j] .= 0.0f0
    @reset st.symbolic_fcns_st[l-1].mask[j, :] .= 0.0f0

    # Remove all outgoing connections
    @reset st.act_fcns_st[l].mask[j, :] .= 0.0f0
    @reset st.symbolic_fcns_st[l].mask[:, j] .= 0.0f0

    return st
end

function prune(rng::AbstractRNG, m, ps, st; threshold=1e-3, mode="auto", active_neurons_id=nothing, verbose=true)
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
    add_to_array!(mask, ones(m.widths[1], ))
    active_neurons_id = [[1:m.widths[1]...]]

    for i in 1:m.depth-1
        if mode == "auto"
            in_important = ifelse.(maximum(st.act_scale[i, :, :], dims=2)[1, :, :] .> threshold, 1.0, 0.0)
            out_important = ifelse.(maximum(st.act_scale[i+1, :, :], dims=1)[1, :, :] .> threshold, 1.0, 0.0)
            overall_important = in_important .* out_important
        elseif mode == "manual"
            overall_important = zeros(Float32, m.widths[i+1])
            overall_important[active_neurons_id[i+1]] .= 1.0
        end

        push!(mask, overall_important)
        cart_ind = findall(x -> x == true, overall_important)
        push!(active_neurons_id, [i[1] for i in cart_ind])
    end
    
    push!(active_neurons_id, [1:m.widths[end]...])
    push!(mask, ones(Float32, m.widths[end], ))

    for i in 1:m.depth-1
        for j in 1:m.widths[i+1]
            if !(j in active_neurons_id[i+1])
                new_st = remove_node(st, i+1, j; verbose=verbose)
            end
        end
    end

    model_pruned = KAN_model(deepcopy(m.widths); k=m.degree, grid_interval=m.grid_interval, ε_scale=m.ε_scale, μ_scale=m.μ_scale, σ_scale=m.σ_scale, base_act=m.base_fcn, symbolic_enabled=m.symbolic_enabled, grid_eps=m.grid_eps, grid_range=m.grid_range)

    ps_pruned = Lux.initialparameters(rng, model_pruned)
    st_pruned = Lux.initialstates(rng, model_pruned)

    for i in 1:size(st.act_scale, 1)
        if i < size(st.act_scale, 1) - 1
            @reset ps_pruned.biases[Symbol("layer_$i")] = ps.biases[Symbol("layer_$i")][:, active_neurons_id[i+1]]
        end

        new_fcn, ps_new, st_new = get_subset(m.act_fcns[i], ps.act_fcns_ps[Symbol("layer_$i")], st.act_fcns_st[i], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.act_fcns[i] = new_fcn
        @reset ps_pruned.act_fcns_ps[Symbol("layer_$i")] = ps_new
        @reset st_pruned.act_fcns_st[i] = st_new

        new_fcn, ps_new, st_new = get_symb_subset(m.symbolic_fcns[i], ps.symbolic_fcns_ps[Symbol("layer_$i")], st.symbolic_fcns_st[i], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.symbolic_fcns[i] = new_fcn
        @reset ps_pruned.symbolic_fcns_ps[Symbol("layer_$i")] = ps_new
        @reset st_pruned.symbolic_fcns_st[i] = st_new
    end

    return model_pruned, ps_pruned, st_pruned
end

@nograd function update_grid(model, x, ps, st)
    """
    Update the grid for each b-spline layer in the model.
    """

    act_fcns_ps_arr = []
    acts_fcns_st = []
 
    for i in 1:model.depth
        _, st = model(x, ps, st)
        new_l, new_ps, new_st = update_lyr_grid(model.act_fcns[i], ps.act_fcns_ps[Symbol("layer_$i")], st.act_fcns_st[i], st.acts[i])
        @reset model.act_fcns[i] = new_l
        push!(act_fcns_ps_arr, new_ps)
        push!(acts_fcns_st, new_st)
    end

    for i in eachindex(act_fcns_ps_arr)
        @reset ps.act_fcns_ps[Symbol("layer_$i")] = act_fcns_ps_arr[i]
    end

    updated_st = (
        act_fcns_st=acts_fcns_st,
        symbolic_fcns_st=st.symbolic_fcns_st,
        acts = st.acts,
        pre_acts = st.pre_acts,
        post_acts = st.post_acts,
        post_splines = st.post_splines,
        act_scale = st.act_scale,
        mask = st.mask,
    )
        
    return model, ps, updated_st
end
    
end