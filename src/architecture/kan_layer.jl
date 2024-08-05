module dense_kan

export KAN_Dense, kan_dense, update_lyr_grid, get_subset

using CUDA, KernelAbstractions
using Lux, Tullio, NNlib, Random, Accessors

include("spline.jl")
include("../utils.jl")
using .Spline: extend_grid, coef2curve, curve2coef
using .Utils: sparse_mask, device

struct kan_dense <: Lux.AbstractExplicitLayer
    in_dim::Int
    out_dim::Int
    num_splines::Int
    degree::Int
    grid::AbstractArray{Float32}
    RBF_σ::Float32
    base_act
    grid_eps::Float32
    grid_range::Tuple{Float32, Float32}
    ε_scale::Float32
    σ_base::Matrix{Float32}
    σ_sp::Float32
end

function KAN_Dense(in_dim::Int, out_dim::Int; num_splines=5, degree=3, ε_scale=0.1, σ_base=nothing, σ_sp=1.0, base_act=NNlib.selu, grid_eps=0.02, grid_range=(-1, 1))
    grid = range(grid_range[1], grid_range[2], length=num_splines + 1) |> collect |> x -> reshape(x, 1, length(x))
    grid = repeat(grid, in_dim, 1) 
    grid = extend_grid(grid, degree) |> device
    init_σ = 1.0

    σ_base = isnothing(σ_base) ? ones(Float32, in_dim, out_dim) : σ_base
    
    return kan_dense(in_dim, out_dim, num_splines, degree, grid, init_σ, base_act, grid_eps, grid_range, ε_scale, σ_base, σ_sp)
end

function Lux.initialparameters(rng::AbstractRNG, l::kan_dense)
    ε = ((rand(rng, Float32, l.num_splines + 1, l.in_dim, l.out_dim) .- 0.5f0) .* l.ε_scale ./ l.num_splines) |> device
    coef = curve2coef(l.grid[:, l.degree+1:end-l.degree] |> permutedims, ε, l.grid; k=l.degree, scale=l.RBF_σ)
    
    w_base = ones(Float32, l.in_dim, l.out_dim) .* l.σ_base
    w_sp = ones(Float32, l.in_dim, l.out_dim) .* l.σ_sp

    return (ε=ε, coef=coef, w_base=w_base, w_sp=w_sp)
end

function Lux.initialstates(rng::AbstractRNG, l::kan_dense)
    sparse_init = parse(Bool, get(ENV, "SPARSE_INIT", "false"))

    if sparse_init
        mask = sparse_mask(l.in_dim, l.out_dim)
    else
        mask = ones(Float32, l.in_dim, l.out_dim)
    end

    return (mask=mask, pre_acts=nothing, post_acts=nothing, post_spline=nothing)
end

function (l::kan_dense)(x, ps, mask)
    """
    Forward pass of the KAN layer.

    Args:
        l: The kan_dense layer.
        ps: The parameters of the layer.
        mask: The mask of the layer.
        x: A matrix of size (b, d) containing the input data.

    Returns:
        y: A matrix of size (b, o) containing the post_activation output data.
        new_st: The updated state of the layer.
    """
    b_size = size(x, 1)

    pre_acts = repeat(reshape(copy(x), b_size, 1, l.in_dim), 1, l.out_dim, 1)
    base = l.base_act(x) # b(x)

    # B-spline basis functions of degree k
    y = coef2curve(x, l.grid, ps.coef; k=l.degree, scale=l.RBF_σ) # spline(x)
    post_spline = permutedims(copy(y), [1, 3, 2])

    # w_b*b(x) + w_s*spline(x)
    y = @tullio out[b, i, o] := (ps.w_base[i, o] * base[b, i] + ps.w_sp[i, o] * y[b, i, o]) * mask[i, o]
    post_acts = permutedims(copy(y), [1, 3, 2])

    # Inner Kolmogorov-Arnold sum
    y = sum(y, dims=2)[:, 1, :]

    new_st = (mask=mask, pre_acts=pre_acts, post_acts=post_acts, post_spline=post_spline)
    return y, new_st
end


function update_lyr_grid(l, ps, x)
    """
    Adapt the grid to the distribution of the input data

    Args:
        l: The kan_dense layer.
        ps: The parameters of the layer.
        st: The state of the layer.
        x: A matrix of size (b, d) containing the input data.

    Returns:
        l: The updated kan_dense layer.
        ps: The updated parameters.
    """
    b_size = size(x, 1)
    
    # Compute the B-spline basis functions of degree k
    CUDA.@allowscalar x_sort = sortslices(x, dims=1)
    current_splines = coef2curve(x_sort, l.grid, ps.coef; k=l.degree, scale=l.RBF_σ)

    # Adaptive grid - concentrate grid points around regions of higher density
    num_interval = size(l.grid, 2) - 2*l.degree - 1
    ids = [div(b_size * i, num_interval) + 1 for i in 0:num_interval-1]
    grid_adaptive = zeros(Float32, 0, size(x, 2)) |> device
    for idx in ids
        grid_adaptive = vcat(grid_adaptive, x_sort[idx:idx, :])
    end
    grid_adaptive = vcat(grid_adaptive, x_sort[end:end, :])
    grid_adaptive = grid_adaptive |> permutedims 

    # Uniform grid
    h = (grid_adaptive[:, end:end] .- grid_adaptive[:, 1:1]) ./ num_interval # step size
    range = collect(0:num_interval)[:, :] |> permutedims |> device
    grid_uniform = h .* range .+ grid_adaptive[:, 1:1] 

    # Grid is a convex combination of the uniform and adaptive grid
    grid = l.grid_eps .* grid_uniform + (1 - l.grid_eps) .* grid_adaptive
    new_grid = extend_grid(grid, l.degree)
    new_coef = curve2coef(x_sort, current_splines, new_grid; k=l.degree, scale=l.RBF_σ)
    
    # Update parameters
    @reset ps.coef = new_coef
    @reset l.grid = new_grid

    return l, ps
end

function get_subset(l, ps, old_mask, in_indices, out_indices)
    """
    Extract smaller subset of the layer for pruning.

    Args:
        l: The kan_dense layer.
        ps: The parameters of the layer.
        old_mask: The mask of the layer.
        in_indices: The indices of the input neurons to keep.
        out_indices: The indices of the output neurons to keep.

    Returns:
        l_subset: The subset kan_dense layer.
        ps_subset: The subset parameters.
        new_mask: The new mask.
    """
    
    l_sub = KAN_Dense(l.in_dim, l.out_dim;
        num_splines=l.num_splines,
        degree=l.degree,
        ε_scale=l.ε_scale,
        σ_base=l.σ_base[in_indices, out_indices],
        σ_sp=l.σ_sp,
        base_act=l.base_act,
        grid_eps=l.grid_eps,
        grid_range=l.grid_range
    )

    @reset l_sub.in_dim = length(in_indices)
    @reset l_sub.out_dim = length(out_indices)
    @reset l_sub.grid = l.grid[in_indices, :]

    # Initialize new parameters
    ps_sub = (
        ε = ps.ε[:, in_indices, out_indices],
        coef = ps.coef[in_indices, out_indices, :],
        w_base = ps.w_base[in_indices, out_indices],
        w_sp = ps.w_sp[in_indices, out_indices]
    )

    return l_sub, ps_sub, old_mask[in_indices, out_indices]
end

end