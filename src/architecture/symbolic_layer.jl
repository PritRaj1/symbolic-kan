module symbolic_layer

export symbolic_kan_layer, lock_symbolic!, get_symb_subset, symb_fwd
using Zygote: @nograd

using Flux, Tullio, Random
# using CUDA, KernelAbstractions
using FunctionWrappers: FunctionWrapper

include("../symbolic_lib.jl")
include("../utils.jl")
using .SymbolicLib: SYMBOLIC_LIB
using .Utils: fit_params

mutable struct symbolic_dense
    in_dim::Int
    out_dim::Int
    mask::AbstractArray
    fcns
    fcns_avoid_singular
    fcn_names::Vector{Vector{String}}
    fcn_sympys
    affine::AbstractArray
end

function symbolic_kan_layer(in_dim::Int, out_dim::Int)
    mask = ones(out_dim, in_dim)
    fcns = [[FunctionWrapper{Float64, Tuple{Float64}}(x -> 0.0) for i in 1:in_dim] for j in 1:out_dim] 
    fcns_avoid_singular = [[FunctionWrapper{Tuple{Float64, Float64}, Tuple{Float64, Float64}}(((x), (y_th)) -> (0.0, 0.0)) for i in 1:in_dim] for j in 1:out_dim]
    fcn_names = [["0" for i in 1:in_dim] for j in 1:out_dim]
    fcn_sympys = [[FunctionWrapper{Float64, Tuple{Float64}}(x -> 0.0) for i in 1:in_dim] for j in 1:out_dim] 
    affine = zeros(out_dim, in_dim, 4)

    return symbolic_dense(in_dim, out_dim, mask, fcns, fcns_avoid_singular, fcn_names, fcn_sympys, affine)
end

function apply_fcn(x, y; fcn)
    if !isnothing(y)
        return fcn(x, y)[2]
    else
        return fcn(x)
    end
end

@nograd function symb_fwd(l, x; avoid_singular=false, y_th=10.0)
    """
    Apply symbolic dense layer to input x using Kolmogorov-Arnold theorm.
    
    i.e apply reference univariate functions to each element of multi-dim sample,
        then sum along input dimension.

    Args:
    - l: symbolic dense layer.
    - x: input tensor of shape (batch_size, in_dim).
    - avoid_singular: whether to avoid singularities.
    - y_th: threshold for singularities.

    Returns:
    - z: output tensor of shape (batch_size, out_dim).
    - post_acts: post activation tensor of shape (batch_size, out_dim, in_dim).
    """

    b_size = size(x, 1)
    y_th = avoid_singular ? repeat([y_th], b_size, 1) : nothing
    fcns = avoid_singular ? l.fcns_avoid_singular : l.fcns

    post_acts = zeros(b_size, l.out_dim, 0) 
    for i in 1:l.in_dim
        post_acts_ = zeros(b_size, 0) 
        for j in 1:l.out_dim
            term1 = l.affine[j, i, 1] .* x[:, i:i] .+ l.affine[j, i, 2]
            f_x = apply_fcn.(term1, y_th; fcn=fcns[j][i])
            xij = l.affine[j, i, 3] .* f_x .+ l.affine[j, i, 4]
            post_acts_ = hcat(post_acts_, l.mask[j, i] .* xij)
        end
        post_acts_ = reshape(post_acts_, b_size, l.out_dim, 1)
        post_acts = cat(post_acts, post_acts_, dims=3)
    end

    return sum(post_acts, dims=3)[:, :, 1], post_acts
end

function get_symb_subset(l::symbolic_dense, in_indices, out_indices)
    """
    Extract smaller symbolic dense layer from larger layer for pruning.
    
    Args:
    - l: symbolic dense layer.
    - in_indices: indices of input dimensions to keep.
    - out_indices: indices of output dimensions to keep.

    Returns:
    - l_sub: subset of symbolic dense layer.
    """
    l_sub = symbolic_kan_layer(l.in_dim, l.out_dim)

    l_sub.in_dim, l_sub.out_dim = length(in_indices), length(out_indices)
    
    new_mask = zeros(l_sub.out_dim, 0) 
    for i in in_indices
        new_mask_ = zeros(0, 1) 
            for j in out_indices
                new_mask_ = vcat(new_mask_, l.mask[j:j, i:i])
            end
        new_mask = hcat(new_mask, new_mask_)
    end

    l_sub.mask = new_mask
    l_sub.fcns = [[l.fcns[j][i] for i in in_indices] for j in out_indices]
    l_sub.fcns_avoid_singular = [[l.fcns_avoid_singular[j][i] for i in in_indices] for j in out_indices]
    l_sub.fcn_names = [[l.fcn_names[j][i] for i in in_indices] for j in out_indices]
    l_sub.fcn_sympys = [[l.fcn_sympys[j][i] for i in in_indices] for j in out_indices]
    l_sub.affine = l.affine[out_indices, in_indices, :]

    return l_sub
end

function set_affine!(l::symbolic_dense, j, i; a1=1.0, a2=0.0, a3=1.0, a4=0.0)
    """
    Set affine parameters for symbolic dense layer.
    
    Args:
    - l: symbolic dense layer.
    - j: index of output neuron.
    - i: index of input neuron.
    - a1: param1.
    - a2: param2.
    - a3: param3.
    - a4: param4.
    """
    l.affine[j, i, 1] = a1
    l.affine[j, i, 2] = a2
    l.affine[j, i, 3] = a3
    l.affine[j, i, 4] = a4
end

function lock_symbolic!(l::symbolic_dense, i, j, fun_name; x=nothing, y=nothing, random=false, seed=nothing, α_range=(-10, 10), β_range=(-10, 10), μ=1.0, verbose=true)
    """
    Fix a symbolic function for a particular input-output pair, 
    
    i.e. the univariate fcn used for a particular evaluation of the Kolmogorov-Arnold theorem.

    Args:
    - l: symbolic dense layer.
    - i: index of input neuron.
    - j: index of output neuron.
    - fun_name: name of symbolic function to lock.
    - x: 1D array of preactivations
    - y: 1D array of postactivations
    - random: whether to randomly initialise function parameters.
    - α_range: sweep range for α parameter.
    - β: sweep range for β parameter.
    - verbose: whether to print updates.

    Returns:
    - R2: coefficient of determination.
    """
    # fun_name is a name of a symbolic function
    if fun_name isa String
        fcn = SYMBOLIC_LIB[fun_name][1]
        fcn_sympy = SYMBOLIC_LIB[fun_name][2]
        fcn_avoid_singular = SYMBOLIC_LIB[fun_name][3]

        l.fcn_sympys[j][i] = FunctionWrapper{Float64, Tuple{Float64}}(fcn_sympy)
        l.fcn_names[j][i] = fun_name

        # If x and y are not provided, just set the function
        if isnothing(x) || isnothing(y)
            l.fcns[j][i] = FunctionWrapper{Float64, Tuple{Float64}}(fcn)
            l.fcns_avoid_singular[j][i] = FunctionWrapper{Tuple{Float64, Float64}, Tuple{Float64, Float64}}(fcn_avoid_singular)

            # Set affine parameters either to random values or to default values
            if !random
                set_affine!(l, j, i)
            else
                Random.seed!(seed)
                params = rand(4) .* 2 .- 1
                set_affine!(l, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
            end

        # If x and y are provided, fit the function
        else
            params, R2 = fit_params(x, y, fcn; α_range=α_range, β_range=β_range, μ=μ, verbose=verbose)
            l.fcns[j][i] = FunctionWrapper{Float64, Tuple{Float64}}(fcn)
            l.fcns_avoid_singular[j][i] = FunctionWrapper{Tuple{Float64, Float64}, Tuple{Float64, Float64}}(fcn_avoid_singular)
            set_affine!(l, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
            return R2
        end

    # fun_name is a symbolic function
    else
        l.fcns[j][i] = fun_name
        l.fcn_sympys[j][i] = fun_name
        l.fcns_avoid_singular[j][i] = fun_name
        l.fcn_names[j][i] = "anonymous"

        # Set affine parameters either to random values or to default values
        if !random
            set_affine!(l, j, i)
        else
            Random.seed!(seed)
            params = rand(4) .* 2 .- 1
            set_affine!(l, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
        end
    end

    return nothing
end

end
