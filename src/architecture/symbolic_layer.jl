module symbolic_layer

export symbolic_kan_kayer

include("../symbolic_lib.jl")
include("../utils.jl")

using Flux, CUDA, KernelAbstractions, Tullio
using .SymbolicLib: SYMBOLIC_LIB
using .Utils: device

struct symbolic_dense
    in_dim::Int
    out_dim::Int
    mask
    fcns
    fcns_avoid_singular
    fcn_names
    fcn_sympys
    affine
end

function symbolic_kan_kayer(in_dim::Int, out_dim::Int)
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
    b_size = size(x, 1)
    avoid_singular ? y_th = repeat([10.0], b_size, 1) |> device : y_th = nothing

    post_acts = zeros(b_size, l.out_dim, 0)
    for i in 1:l.in_dim
        post_acts_ = zeros(b_size, 0)
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

end