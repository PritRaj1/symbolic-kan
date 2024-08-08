module symbolic_layer

export SymbolicDense, symbolic_dense, get_symb_subset

using CUDA, KernelAbstractions
using Lux, Tullio, Random, SymPy, Accessors

include("../symbolic_lib.jl")
include("../utils.jl")
using .SymbolicLib: SYMBOLIC_LIB
using .Utils: device

struct symbolic_dense <: Lux.AbstractExplicitLayer
    in_dim::Int
    out_dim::Int
    fcns::Vector{Vector{Function}}
    fcns_avoid_singular::Vector{Vector{Function}}
    fcn_names::Vector{Vector{String}}
    fcn_sympys::Vector{Vector{Union{SymPy.Sym, Function}}}
end

function SymbolicDense(in_dim::Int, out_dim::Int)
    fcns = [[x -> x*0.0f0 for i in 1:in_dim] for j in 1:out_dim] 
    fcns_avoid_singular = [[(x, y_th) -> (x*0.0f0, x*0.0f0) for i in 1:in_dim] for j in 1:out_dim]
    fcn_names = [["0" for i in 1:in_dim] for j in 1:out_dim]
    fcn_sympys = [[x -> x*0.0f0 for i in 1:in_dim] for j in 1:out_dim] 

    return symbolic_dense(in_dim, out_dim, fcns, fcns_avoid_singular, fcn_names, fcn_sympys)
end

function Lux.initialparameters(rng::AbstractRNG, l::symbolic_dense)
    ps = (
        affine = zeros(Float32, l.out_dim, l.in_dim, 4)
    )
    return ps
end

function Lux.initialstates(rng::AbstractRNG, l::symbolic_dense)
    mask = zeros(Float32, l.out_dim, l.in_dim)
    st = (
        mask = mask,
        post_acts = nothing
    )
    return st
end

function apply_fcn(x, y; fcn)
    if !isnothing(y)
        return fcn(x, y)[2]
    else
        return fcn(x)
    end
end

### c * f(a*x + b) + d ###  
function (l::symbolic_dense)(x, ps, mask; avoid_singular=true, y_th=10.0f0)
    """
    Forward pass of the symbolic dense layer evaluated for input x.

    Args:
        x: A matrix of size (b, in_dim) containing the input data.
        ps: A tuple containing the parameters of the layer.
        mask: A matrix of size (out_dim, in_dim) containing the mask of the layer.
        avoid_singular: A boolean indicating whether to avoid singularities.
        y_th: Secondary argument for singular avoiding functions.

    Returns:
        z: A matrix of size (b, out_dim) containing the output data.
        new_st: A tuple containing the new state of the layer.
    """
    b_size = size(x, 1)
    y_th = avoid_singular ? device(repeat([y_th], b_size,)) : nothing
    fcns = avoid_singular ? l.fcns_avoid_singular : l.fcns

    # ps = affine weights, (a, b, c, d)
    A = selectdim(ps, 3, 1)
    B = selectdim(ps, 3, 2)
    C = selectdim(ps, 3, 3)
    D = selectdim(ps, 3, 4)

    inner_term = @tullio out[b, j, i] := A[j, i] * x[b, i] + B[j, i]

    # Major GPU bottleneck - univariate function application
    ŷ = zeros(Float32, b_size, l.out_dim, 0) |> device
    for i in 1:l.in_dim
        ŷ_inner = zeros(Float32, b_size, 0) |> device
        for j in 1:l.out_dim
            x_eval = selectdim(selectdim(inner_term, 2, j), 2, i)
            f_x = apply_fcn(x_eval, y_th; fcn=fcns[j][i])
            f_x = reshape(f_x, b_size, 1)
            ŷ_inner = cat(ŷ_inner, f_x, dims=2)
        end
        ŷ = cat(ŷ, ŷ_inner, dims=3)
    end

    post_acts = @tullio out[b, j, i] := C[j, i] * ŷ[b, j, i] + D[j, i]
    post_acts = @tullio out[b, j, i] := mask[j, i] * post_acts[b, j, i]

    z = sum(post_acts, dims=3)[:, :, 1]
    new_st = (mask=mask, post_acts=post_acts)
    return z, new_st
end

function get_symb_subset(l::symbolic_dense, ps, old_mask, in_indices, out_indices)
    """
    Extract a subset of the symbolic dense layer.

    Args:
        l: The symbolic dense layer.
        ps: The parameters of the layer.
        old_mask: The mask of the layer.
        in_indices: The indices of the input dimensions to extract.
        out_indices: The indices of the output dimensions to extract.

    Returns:
        l_sub: The subset of the symbolic dense layer.
        ps_sub: The subset of the parameters of the layer.
        mask_sub: The subset of the mask of the layer.
    """
    l_sub = SymbolicDense(l.in_dim, l.out_dim)
    @reset l_sub.in_dim = length(in_indices)
    @reset l_sub.out_dim = length(out_indices)
    
    ps_sub = (
        affine = ps[out_indices, in_indices, :]
    )

    @reset l_sub.fcns = [[l.fcns[j][i] for i in in_indices] for j in out_indices]
    @reset l_sub.fcns_avoid_singular = [[l.fcns_avoid_singular[j][i] for i in in_indices] for j in out_indices]
    @reset l_sub.fcn_names = [[l.fcn_names[j][i] for i in in_indices] for j in out_indices]
    @reset l_sub.fcn_sympys = [[l.fcn_sympys[j][i] for i in in_indices] for j in out_indices]

    return l_sub, ps_sub, old_mask[out_indices, in_indices]
end

end