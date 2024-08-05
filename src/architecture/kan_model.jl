module KolmogorovArnoldNets

export KAN, KAN_model, prune, update_grid

using CUDA, KernelAbstractions, Lux, LuxCUDA
using Lux, Tullio, NNlib, Random, Statistics, SymPy, Accessors
using Zygote: @nograd

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
end

function KAN_model(widths; k=3, grid_interval=3, ε_scale=0.1f0, μ_scale=0.0f0, σ_scale=1.0f0, base_act=NNlib.selu, symbolic_enabled=true, grid_eps=1.0f0, grid_range=(-1f0, 1f0))
    depth = length(widths) - 1

    act_fcns = []
    symbolic_fcns = []

    for i in 1:depth
        base_scale = (μ_scale * (1 / √(widths[i])) 
        .+ σ_scale .* (randn(Float32, widths[i], widths[i + 1]) .* 2 .- 1) .* (1 / √(widths[i])))
        
        spline = KAN_Dense(widths[i], widths[i + 1]; num_splines=grid_interval, degree=k, ε_scale=ε_scale, σ_base=base_scale, σ_sp=1.0, base_act=base_act, grid_eps=grid_eps, grid_range=grid_range)
        push!(act_fcns, spline)
        
        symbolic = SymbolicDense(widths[i], widths[i + 1])
        push!(symbolic_fcns, symbolic)
    end

    return KAN(widths, depth, k, grid_interval, base_act, ε_scale, μ_scale, σ_scale, grid_eps, grid_range, symbolic_enabled, act_fcns, symbolic_fcns)
end

function Lux.initialparameters(rng::AbstractRNG, m::KAN)
    # Create one long named tuple
    ps = NamedTuple(Symbol("bias_$i") => zeros(Float32, 1, m.widths[i+1]) for i in 1:m.depth)
    
    for i in 1:m.depth

        # Parameters for the spline layer
        layer_ps = Lux.initialparameters(rng, m.act_fcns[i])
        ε, coef, w_base, w_sp = layer_ps[:ε], layer_ps[:coef], layer_ps[:w_base], layer_ps[:w_sp]
        @reset ps[Symbol("ε_$i")] = ε
        @reset ps[Symbol("coef_$i")] = coef
        @reset ps[Symbol("w_base_$i")] = w_base
        @reset ps[Symbol("w_sp_$i")] = w_sp

        # Parameters for the symbolic layer
        symb_ps = Lux.initialparameters(rng, m.symbolic_fcns[i])
        @reset ps[Symbol("affine_$i")] = symb_ps

    end

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
        mask = [ones(Float32, widths) for widths in m.widths[1:end]],
        symbolic_acts = []
    )

    for i in 1:m.depth
        @reset st[Symbol("act_scale_$i")] = zeros(Float32, maximum(m.widths[i+1]), maximum(m.widths[i]))
    end

    return st
end

@nograd function add_to_array!(arr, x)
    return push!(arr, x)
end

function (m::KAN)(x, ps, st)
    x, ps, st = device(x), device(ps), device(st)

    x_eval = copy(x)
    acts_arr = [x,]
    pre_acts_arr = []
    post_acts_arr = []
    post_splines_arr = []

    for i in 1:m.depth
        # spline(x)
        kan_ps = (
            ε = ps[Symbol("ε_$i")],
            coef = ps[Symbol("coef_$i")],
            w_base = ps[Symbol("w_base_$i")],
            w_sp = ps[Symbol("w_sp_$i")]
        )
        x_numerical, spline_st = m.act_fcns[i](x_eval, kan_ps, st.act_fcns_st[i])
        @reset st.act_fcns_st[i] = spline_st
        
        # Evaluate symbolic layer at x
        x_symbolic, symbolic_st = Float32(0.0), Float32(0.0)
        if m.symbolic_enabled
            symb_ps = (
                affine = ps[Symbol("affine_$i")]
            )
            x_symbolic, symbolic_st = m.symbolic_fcns[i](x_eval, affine, st.symbolic_fcns_st[i])
            @reset st.symbolic_fcns_st[i] = symbolic_st
        end

        x_eval = x_numerical .+ x_symbolic
        post_acts = spline_st.post_acts .+ symbolic_st.post_acts

        # Scales for l1 regularisation
        in_range = std(spline_st.pre_acts, dims=1) .+ 0.1
        out_range = std(post_acts, dims=1)
        @reset st[Symbol("act_scale_$i")] = (out_range ./ in_range)[1, :, :]
        
        add_to_array!(pre_acts_arr, spline_st.pre_acts)
        add_to_array!(post_acts_arr, post_acts)
        add_to_array!(post_splines_arr, spline_st.post_spline)

        # Add bias b(x)
        b = repeat(ps[Symbol("bias_$i")], size(x_eval, 1), 1)
        x_eval = x_eval + b

        add_to_array!(acts_arr, copy(x_eval))
    end

    @reset st.acts = acts_arr
    @reset st.pre_acts = pre_acts_arr
    @reset st.post_acts = post_acts_arr
    @reset st.post_splines = post_splines_arr

    return x_eval, st
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
    @reset st.act_fcns_st[l-1].mask[:, j] .= 0.0f0
    @reset st.symbolic_fcns_st[l-1].mask[j, :] .= 0.0f0

    # Remove all outgoing connections
    @reset st.act_fcns_st[l].mask[j, :] .= 0.0f0
    @reset st.symbolic_fcns_st[l].mask[:, j] .= 0.0f0

    return st
end

function prune(rng::AbstractRNG, m, ps, st; threshold=0.01, mode="auto", active_neurons_id=nothing, verbose=true)
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


    mask = []
    add_to_array!(mask, ones(m.widths[1], ))
    active_neurons_id = []
    add_to_array!(active_neurons_id, [1:m.widths[1]...])
    overall_important = nothing

    for i in 1:m.depth-1
        if mode == "auto"
            scale1 = st[Symbol("act_scale_$i")]
            scale2 = st[Symbol("act_scale_$(i+1)")]
            in_important = ifelse.(maximum(scale1, dims=2)[1, :, :] .> threshold, Float32(1),  Float32(0))
            out_important = ifelse.(maximum(scale2, dims=1)[1, :, :] .> threshold,  Float32(1),  Float32(0))
            overall_important = in_important .* out_important
        elseif mode == "manual"
            overall_important = zeros(Float32, m.widths[i+1])
            overall_important[active_neurons_id[i+1]] .= 1.0
        end

        add_to_array!(mask, overall_important)
        cart_ind = findall(x -> x > 0.0, overall_important)
        add_to_array!(active_neurons_id, [i[1] for i in cart_ind])
    end
    
    add_to_array!(active_neurons_id, [1:m.widths[end]...])
    add_to_array!(mask, ones(Float32, m.widths[end], ))
    @reset st.mask = mask

    if verbose
        println("Active neurons: ", active_neurons_id)
    end

    for i in 1:m.depth
        for j in 1:m.widths[i+1]
            if !(j in active_neurons_id[i+1])
                st = remove_node(st, i+1, j; verbose=verbose)
            end
        end
    end

    model_pruned = KAN_model(deepcopy(m.widths); k=m.degree, grid_interval=m.grid_interval, ε_scale=m.ε_scale, μ_scale=m.μ_scale, σ_scale=m.σ_scale, base_act=m.base_fcn, symbolic_enabled=m.symbolic_enabled, grid_eps=m.grid_eps, grid_range=m.grid_range)

    ps_pruned = Lux.initialparameters(rng, model_pruned)
    st_pruned = Lux.initialstates(rng, model_pruned)

    for i in 1:m.depth
        if i < m.depth
            @reset ps_pruned[Symbol("bias_$i")] = ps[Symbol("bias_$i")][:, active_neurons_id[i+1]]
        end

        kan_ps = (
            ε = ps_pruned[Symbol("ε_$i")],
            coef = ps_pruned[Symbol("coef_$i")],
            w_base = ps_pruned[Symbol("w_base_$i")],
            w_sp = ps_pruned[Symbol("w_sp_$i")]
        )

        symb_ps = (
            affine = ps_pruned[Symbol("affine_$i")]
        )

        new_fcn, ps_new, st_new = get_subset(model_pruned.act_fcns[i], kan_ps, st_pruned.act_fcns_st[i], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.act_fcns[i] = new_fcn
        ε, coef, w_base, w_sp = ps_new[:ε], ps_new[:coef], ps_new[:w_base], ps_new[:w_sp]
        @reset ps_pruned[Symbol("ε_$i")] = ε
        @reset ps_pruned[Symbol("coef_$i")] = coef
        @reset ps_pruned[Symbol("w_base_$i")] = w_base
        @reset ps_pruned[Symbol("w_sp_$i")] = w_sp
        @reset st_pruned.act_fcns_st[i] = st_new

        new_fcn, ps_new, st_new = get_symb_subset(m.symbolic_fcns[i], symb_ps, st_pruned.symbolic_fcns_st[i], active_neurons_id[i], active_neurons_id[i+1])
        @reset model_pruned.symbolic_fcns[i] = new_fcn
        @reset ps_pruned[Symbol("affine_$i")] = ps_new
        @reset st_pruned.symbolic_fcns_st[i] = st_new

        @reset model_pruned.widths[i] = length(active_neurons_id[i])
    end

    @reset st_pruned.mask = mask
    @reset model_pruned.depth = length(model_pruned.widths) - 1

    return model_pruned, device(ps_pruned), device(st_pruned)
end

@nograd function update_grid(model, x, ps, st)
    """
    Update the grid for each b-spline layer in the model.
    """

    act_fcns_ps_arr = []
    acts_fcns_st = []
 
    for i in 1:model.depth
        _, st = model(x, ps, st)

        kan_ps = (
            ε = ps[Symbol("ε_$i")],
            coef = ps[Symbol("coef_$i")],
            w_base = ps[Symbol("w_base_$i")],
            w_sp = ps[Symbol("w_sp_$i")]
        )

        new_l, new_ps, new_st = update_lyr_grid(model.act_fcns[i], kan_ps, st.act_fcns_st[i], st.acts[i])
        @reset model.act_fcns[i] = new_l
        @reset ps[Symbol("ε_$i")] = new_ps.ε
        @reset ps[Symbol("coef_$i")] = new_ps.coef
        @reset ps[Symbol("w_base_$i")] = new_ps.w_base
        @reset ps[Symbol("w_sp_$i")] = new_ps.w_sp
        push!(acts_fcns_st, new_st)
    end

    updated_st = (
        act_fcns_st=acts_fcns_st,
        symbolic_fcns_st=st.symbolic_fcns_st,
        acts = st.acts,
        pre_acts = st.pre_acts,
        post_acts = st.post_acts,
        post_splines = st.post_splines,
        mask = st.mask,
    )
        
    return model, ps, updated_st
end
    
end