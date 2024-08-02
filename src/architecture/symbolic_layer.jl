module symbolic_layer

export SymbolicDense, symbolic_dense, get_symb_subset

using Lux, Tullio, Random, SymPy
# using CUDA, KernelAbstractions

include("../symbolic_lib.jl")
using .SymbolicLib: SYMBOLIC_LIB

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

function (l::symbolic_dense)(x, ps, st; avoid_singular=true, y_th=10.0f0)
    b_size = size(x, 1)
    y_th = avoid_singular ? repeat([y_th], b_size, 1) : nothing
    fcns = avoid_singular ? l.fcns_avoid_singular : l.fcns

    post_acts = zeros(Float32, b_size, l.out_dim, 0) 

    for i in 1:l.in_dim
        post_acts_ = zeros(Float32, b_size, 0) 
        for j in 1:l.out_dim
            term1 = ps[j, i, 1] .* x[:, i:i] .+ ps[j, i, 2]
            f_x = apply_fcn.(term1, y_th; fcn=fcns[j][i])
            xij = ps[j, i, 3] .* f_x .+ ps[j, i, 4]
            post_acts_ = hcat(post_acts_, st.mask[j, i] .* xij)
        end
        post_acts_ = reshape(post_acts_, b_size, l.out_dim, 1)
        post_acts = cat(post_acts, post_acts_, dims=3)
    end

    z = sum(post_acts, dims=3)[:, :, 1]
    new_st = (mask=st.mask, post_acts=post_acts)
    return z, new_st
end

function get_symb_subset(l::symbolic_dense, ps, st, in_indices, out_indices)
    l_sub = SymbolicDense(length(in_indices), length(out_indices))
    
    ps_sub = (
        affine = ps[out_indices, in_indices, :]
    )

    st_sub = (
        mask = st.mask[out_indices, in_indices],
        post_acts = nothing
    )

    l_sub = symbolic_dense(
        length(in_indices),
        length(out_indices),
        [[l.fcns[j][i] for i in in_indices] for j in out_indices],
        [[l.fcns_avoid_singular[j][i] for i in in_indices] for j in out_indices],
        [[l.fcn_names[j][i] for i in in_indices] for j in out_indices],
        [[l.fcn_sympys[j][i] for i in in_indices] for j in out_indices]
    )

    return l_sub, ps_sub, st_sub
end

end