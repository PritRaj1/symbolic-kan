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
    mask = ones(in_dim, out_dim)
    fcns = [[x -> 0.0 for i in 1:in_dim] for j in 1:out_dim] 
    fcns_avoid_singular = [[(x, y_th) -> ((), x * 0.0) for i in 1:in_dim] for j in 1:out_dim]
    fcn_names = [["0" for i in 1:in_dim] for j in 1:out_dim]
    fcn_sympys = [[x -> 0.0 for i in 1:in_dim] for j in 1:out_dim] 
    affine = zeros(in_dim, out_dim, 4)

    return symbolic_dense(in_dim, out_dim, mask, fcns, fcns_avoid_singular, fcn_names, fcn_sympys, affine)
end

Flux.@functor symbolic_dense (mask, affine)

function (l::symbolic_dense)(x; avoid_singular=false, y_th=10.0)
    b_size = size(x, 1)
    post_acts = zeros(b_size, l.in_dim, 0)
    avoid_singular ? y_th = repeat([10.0], b_size, 1) |> device : y_th = nothing
    # println(size(x), size(y_th))

    for i in 1:l.out_dim
        post_acts_ = zeros(b_size, 0)
        for j in 1:l.in_dim
            if avoid_singular
                f_xy = l.fcns_avoid_singular[j][i].(l.affine[j, i, 1] .* x[:, i:i] .+ l.affine[j, i, 2], y_th)[2]
                xij = l.affine[j, i, 3] .* f_xy .+ l.affine[j, i, 4]
            else
                f_x = l.fcns[j][i].(l.affine[j, i, 1] .* x[:, i:i] .+ l.affine[j, i, 2])
                xij = l.affine[j, i, 3] .* f_x .+ l.affine[j, i, 4]
            end
            post_acts_ = hcat(post_acts_, l.mask[j, i] .* xij)
        end
        println(size(post_acts), " ", size(post_acts_))
        post_acts_ = reshape(post_acts_, b_size, l.in_dim, 1)
        println(size(post_acts), " ", size(post_acts_))
        post_acts = cat(post_acts, post_acts_, dims=3)
    end
    println(size(post_acts))
end

sb = symbolic_kan_kayer(3, 3)
x=randn(100,3)
y = sb(x)
println(size(y))

"""
>>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
"""