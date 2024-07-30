module dense_kan

export b_spline_layer, fwd, update_lyr_grid!, get_subset

using Flux, Tullio, NNlib
# using CUDA, KernelAbstractions

include("spline.jl")
include("../utils.jl")
using .Spline: extend_grid, coef2curve, curve2coef
using .Utils: sparse_mask

mutable struct kan_dense
    in_dim::Int
    out_dim::Int
    num_splines::Int
    degree::Int
    grid::AbstractArray{Float32}
    RBF_σ::Float32
    ε::AbstractArray{Float32}
    coef::AbstractArray{Float32}
    w_base::AbstractArray{Float32}
    w_sp::AbstractArray{Float32}
    base_act
    mask::AbstractArray{Float32}
    grid_eps::Float32
    grid_range::Tuple{Float32, Float32}
end

function b_spline_layer(in_dim::Int, out_dim::Int; num_splines=5, degree=3, ε_scale=0.1, σ_base=1.0, σ_sp=1.0, base_act=NNlib.selu, grid_eps=0.02, grid_range=(-1, 1), sparse_init=false)
    grid = range(grid_range[1], grid_range[2], length=num_splines + 1) |> collect |> x -> reshape(x, 1, length(x))
    grid = repeat(grid, in_dim, 1) 
    grid = extend_grid(grid, degree) 
    
    init_σ = 1.0
    ε = ((rand(num_splines + 1, in_dim, out_dim) .- 0.5) .* ε_scale ./ num_splines)  
    coef = curve2coef(grid[:, degree+1:end-degree] |> permutedims, ε, grid; k=degree, scale=init_σ)
    
    if sparse_init
        mask = sparse_mask(in_dim, out_dim)
    else
        mask = 1.0
    end

    w_base = ones(in_dim, out_dim) .* σ_base .* mask
    w_sp = ones(in_dim, out_dim) .* σ_sp .* mask
    mask = ones(in_dim, out_dim) 

    return kan_dense(in_dim, out_dim, num_splines, degree, grid, init_σ, ε, coef, w_base, w_sp, base_act, mask, grid_eps, grid_range)
end

Flux.@functor kan_dense (ε, coef, w_base, w_sp)

function fwd(l, x)
    b_size = size(x, 1)

    # Base activation.
    pre_acts = repeat(reshape(copy(x), b_size, 1, l.in_dim), 1, l.out_dim, 1)
    base = l.base_act(x)

    # B-spline basis functions of degree k
    y = coef2curve(x, l.grid, l.coef; k=l.degree, scale=l.RBF_σ)
    post_spline = permutedims(copy(y), [1, 3, 2])

    # w_b*b(x) + w_s*spline(x).
    y = @tullio out[b, i, o] := (l.w_base[i, o] * base[b, i] + l.w_sp[i, o] * y[b, i, o]) * l.mask[i, o]
    post_acts = permutedims(copy(y), [1, 3, 2])

    y = sum(y, dims=2)[:, 1, :]

    return y, pre_acts, post_acts, post_spline
end

function update_lyr_grid!(l, x; margin=0.01)
    """
    Adapt the grid to the distribution of the input data

    Args:
        l: The KAN layer.
        x: A matrix of size (b, d) containing the input data.
        y: A matrix of size (b, l) containing the target data.
        margin: The margin to add to the grid.
    """
    b_size = size(x, 1)
    
    # Compute the B-spline basis functions of degree k
    x_sort = sortslices(x, dims=1)
    println("x_sort: ", size(x_sort))
    current_splines = coef2curve(x_sort, l.grid, l.coef; k=l.degree, scale=l.RBF_σ)

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(l.grid, 2) - 2*l.degree - 1
    ids = [div(b_size * i, num_interval) + 1 for i in 0:num_interval-1]
    grid_adaptive = zeros(Float32, 0, size(x, 2)) 
    for idx in ids
        grid_adaptive = vcat(grid_adaptive, x_sort[idx:idx, :])
    end
    grid_adaptive = vcat(grid_adaptive, x_sort[end:end, :])
    grid_adaptive = grid_adaptive |> permutedims 

    # Uniform grid
    h = (grid_adaptive[:, end:end] .- grid_adaptive[:, 1:1]) ./ num_interval # step size
    range = collect(0:num_interval)[:, :] |> permutedims 
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1] 

    # Grid is a convex combination of the uniform and adaptive grid
    grid = @tullio out[i, j] := l.grid_eps * grid_uniform[i, j] + (1 - l.grid_eps) * grid_adaptive[i, j]
    println("grid: ", grid)
    l.grid = extend_grid(grid, l.degree)
    println("l.grid: ", l.grid)
    l.coef = curve2coef(x_sort, current_splines, l.grid; k=l.degree, scale=l.RBF_σ)
    println("l.coef: ", l.coef)
end

function get_subset(l::kan_dense, in_indices, out_indices)
    """
    Extract smaller subset of the layer for pruning.

    Args:
        l: The KAN layer.
        in_indices: The indices of the input neurons to keep.
        out_indices: The indices of the output neurons to keep.

    Returns:
        l_subset: The subset KAN layer.
    """
    l_sub = b_spline_layer(l.in_dim, l.out_dim; num_splines=l.num_splines, degree=l.degree, ε_scale=0.1, σ_base=1.0, σ_sp=1.0, base_act=l.base_act, grid_eps=l.grid_eps, grid_range=l.grid_range, sparse_init=false)
    l_sub.in_dim = length(in_indices)
    l_sub.out_dim = length(out_indices)

    new_grid = zeros(Float32, 0, size(l.grid, 2)) 
    for i in in_indices
        new_grid = vcat(new_grid, l.grid[i:i, :])
    end
    l_sub.grid = new_grid

    l_sub.ε = zeros(Float32, size(l.ε, 1), l_sub.in_dim, l_sub.out_dim) 
    l_sub.coef = zeros(Float32, l_sub.in_dim, l_sub.out_dim, size(l.coef, 3)) 
    l_sub.w_base = zeros(Float32, l_sub.in_dim, l_sub.out_dim) 
    l_sub.w_sp = zeros(Float32, l_sub.in_dim, l_sub.out_dim) 
    l_sub.mask = zeros(Float32, l_sub.in_dim, l_sub.out_dim) 
    
    for in_idx in eachindex(in_indices)
        for out_idx in eachindex(out_indices)
            i, j = in_indices[in_idx], out_indices[out_idx]

            l_sub.ε[:, in_idx, out_idx] .= l.ε[:, i, j]
            l_sub.coef[in_idx, out_idx, :] .= l.coef[i, j, :]
            l_sub.w_base[in_idx, out_idx] = l.w_base[i, j]
            l_sub.w_sp[in_idx, out_idx] = l.w_sp[i, j]
            l_sub.mask[in_idx, out_idx] = l.mask[i, j]
        end
    end

    return l_sub
end

end