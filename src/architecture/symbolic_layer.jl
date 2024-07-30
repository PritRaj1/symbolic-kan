module symbolic_layer

export symbolic_kan_layer, get_symb_subset, symb_fwd

using Flux, Tullio, Random, SymPy
# using CUDA, KernelAbstractions

include("../symbolic_lib.jl")
using .SymbolicLib: SYMBOLIC_LIB

mutable struct symbolic_dense
    in_dim::Int
    out_dim::Int
    mask::AbstractArray{Float32}
    fcns::Vector{Vector{Function}}
    fcns_avoid_singular::Vector{Vector{Function}}
    fcn_names::Vector{Vector{String}}
    fcn_sympys::Vector{Vector{Union{SymPy.Sym, Function}}}
    affine::AbstractArray{Float32}
end

function symbolic_kan_layer(in_dim::Int, out_dim::Int)
    mask = zeros(out_dim, in_dim)
    fcns = [[x -> x*0.0f0 for i in 1:in_dim] for j in 1:out_dim] 
    fcns_avoid_singular = [[(x, y_th) -> (x*0.0f0, x*0.0f0) for i in 1:in_dim] for j in 1:out_dim]
    fcn_names = [["0" for i in 1:in_dim] for j in 1:out_dim]
    fcn_sympys = [[x -> x*0.0f0 for i in 1:in_dim] for j in 1:out_dim] 
    affine = zeros(out_dim, in_dim, 4)

    return symbolic_dense(in_dim, out_dim, mask, fcns, fcns_avoid_singular, fcn_names, fcn_sympys, affine)
end

Flux.@functor symbolic_dense (affine,)

function apply_fcn(x, y; fcn)
    if !isnothing(y)
        return fcn(x, y)[2]
    else
        return fcn(x)
    end
end

function symb_fwd(l, x; avoid_singular=true, y_th=10.0)
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

    post_acts = zeros(Float32, b_size, l.out_dim, 0) 
    for i in 1:l.in_dim
        post_acts_ = zeros(Float32, b_size, 0) 
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

function get_symb_subset(l, in_indices, out_indices)
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

    l_sub.in_dim = length(in_indices)
    l_sub.out_dim = length(out_indices)
    
    new_mask = zeros(Float32, l_sub.out_dim, 0) 
    for i in in_indices
        new_mask_ = zeros(Float32, 0, 1) 
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
