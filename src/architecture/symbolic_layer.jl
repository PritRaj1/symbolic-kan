module symbolic_layer

export symbolic_kan_layer, get_subset

using Flux, CUDA, KernelAbstractions, Tullio

include("../symbolic_lib.jl")
include("../utils.jl")
using .SymbolicLib: SYMBOLIC_LIB
using .Utils: device

mutable struct symbolic_dense
    in_dim::Int
    out_dim::Int
    mask
    fcns
    fcns_avoid_singular
    fcn_names
    fcn_sympys
    affine
end

function symbolic_kan_layer(in_dim::Int, out_dim::Int)
    mask = ones(out_dim, in_dim)
    fcns = [[x -> 0.0 for i in 1:in_dim] for j in 1:out_dim] 
    fcns_avoid_singular = [[(x, y_th) -> ((), x * 0.0) for i in 1:in_dim] for j in 1:out_dim]
    fcn_names = [["0" for i in 1:in_dim] for j in 1:out_dim]
    fcn_sympys = [[x -> 0.0 for i in 1:in_dim] for j in 1:out_dim] 
    affine = zeros(out_dim, in_dim, 4)

    return symbolic_dense(in_dim, out_dim, mask, fcns, fcns_avoid_singular, fcn_names, fcn_sympys, affine)
end

Flux.@functor symbolic_dense (mask, affine)

function (l::symbolic_dense)(x; avoid_singular=false, y_th=10.0)
    """
    Apply symbolic dense layer to input x using Kolmogorov-Arnold theorm.
    
    i.e apply reference univariate functions to each element of multi-dim sample,
        then sum along input dimension.

    Args:
    - x: input tensor of shape (batch_size, in_dim).
    - avoid_singular: whether to avoid singularities.
    - y_th: threshold for singularities.

    Returns:
    - z: output tensor of shape (batch_size, out_dim).
    - post_acts: post activation tensor of shape (batch_size, out_dim, in_dim).
    """

    b_size = size(x, 1)
    avoid_singular ? y_th = repeat([10.0], b_size, 1) |> device : y_th = nothing

    post_acts = zeros(b_size, l.out_dim, 0) |> device
    for i in 1:l.in_dim
        post_acts_ = zeros(b_size, 0) |> device
        for j in 1:l.out_dim
            if avoid_singular
                f_xy = l.fcns_avoid_singular[j][i].(l.affine[j, i, 1] .* x[:, i:i] .+ l.affine[j, i, 2], y_th)[2]
                xij = l.affine[j, i, 3] .* f_xy .+ l.affine[j, i, 4]
            else
                f_x = l.fcns[j][i].(l.affine[j, i, 1] .* x[:, i:i] .+ l.affine[j, i, 2])
                xij = l.affine[j, i, 3] .* f_x .+ l.affine[j, i, 4]
            end
            post_acts_ = hcat(post_acts_, l.mask[j, i] .* xij)
        end
        post_acts_ = reshape(post_acts_, b_size, l.out_dim, 1)
        post_acts = cat(post_acts, post_acts_, dims=3)
    end

    return sum(post_acts, dims=3)[:, :, 1], post_acts
end

function get_subset(l::symbolic_dense, in_indices, out_indices)
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
    
    new_mask = zeros(l_sub.out_dim, 0) |> device
    for i in in_indices
        new_mask_ = zeros(0, 1) |> device
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

end
