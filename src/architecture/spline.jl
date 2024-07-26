module Spline

export extend_grid, B_batch, coef2curve, curve2coef

using Flux, Tullio, LinearAlgebra
using Zygote: @adjoint
# using CUDA, KernelAbstractions

include("../utils.jl")
using .Utils: sparse_mask

method = get(ENV, "METHOD", "spline") # "spline" or "RBF"; RBF not properly implemented yet

function removeNaN(x)
    return isnan(x) ? Float32(0.0) : x
end

function removeZero(x; ε=1e-3)
    return iszero(x) ? Float32(ε) : x
end

function extend_grid(grid, k_extend=0)
    """
    Extend the grid of knots to include the boundary knots.

    Args:
        grid: A matrix of size (d, m) containing the grid of knots.
        k_extend: The number of boundary knots to add to the grid.
    
    Returns:
        A matrix of size (d, m + 2 * k_extend) containing the extended grid of knots.
    """
    h = (grid[ :, end] .- grid[:, 1]) ./ (size(grid, 2) - 1)

    for i in 1:k_extend
        grid = hcat(grid[:, 1:1] .- h, grid)
        grid = hcat(grid, grid[:, end:end] .+ h)
    end
    
    return grid
end

function B_batch(x, grid; degree::Int64, σ=nothing)
    """
    Compute the B-spline basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (d, n) containing the points at which to evaluate the B-spline basis functions.
        grid: A matrix of size (d, m) containing the grid of knots.
        degree: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (d, m, n) containing the B-spline basis functions evaluated at the points x.
    """
    # Expand for broadcasting
    x_eval = repeat(reshape(x, size(x)..., 1), 1, 1, size(grid, 2) - 1)
    grid_eval = repeat(reshape(grid, 1, size(grid)...), size(x, 1), 1, 1)
    
    # B-spline basis functions of degree 0 are piecewise constant functions: B = 1 if x in [grid[p], grid[p+1]) else 0
    grid_1 = grid_eval[:, :, 1:end-1] # grid[p]
    grid_2 = grid_eval[:, :, 2:end] # grid[p+1]

    term1 = ifelse.(x_eval .>= grid_1, 1.0, 0.0)
    term2 = ifelse.(x_eval .< grid_2, 1.0, 0.0)
    term1 |> collect 
    term2 |> collect 

    B = @tullio res[d, p, n] := term1[d, p, n] * term2[d, p, n]

    x = reshape(x, size(x)..., 1) 
    grid = reshape(grid, 1, size(grid)...) 

    # Compute the B-spline basis functions of degree k
    for k in 1:degree 
        numer1 = x .- grid[:, :, 1:(end - k - 1)]
        denom1 = grid[:, :, (k + 1):end-1] .- grid[:, :, 1:(end - k - 1)]
        numer2 = grid[:, :, (k + 2):end] .- x
        denom2 = grid[:, :, (k + 2):end] .- grid[:, :, 2:(end - k)]
        B_i1 = B[:, :, 1:end - 1]
        B_i2 = B[:, :, 2:end]
        B = @tullio out[d, n, m] := (numer1[d, n, m] / denom1[1, n, m] * B_i1[d, n, m]) + (numer2[d, n, m] / denom2[1, n, m] * B_i2[d, n, m])
    end

    B = removeNaN.(B)
    return B 
end

function B_batch_RBF(x, grid; degree=nothing, σ=1.0)
    """
    Compute the B-spline basis functions for a batch of points x and a grid of knots using the RBF kernel.

    Args:
        x: A matrix of size (d, n) containing the points at which to evaluate the B-spline basis functions.
        grid: A matrix of size (d, m) containing the grid of knots.
        sigma: The bandwidth of the RBF kernel.

    Returns:
        A matrix of size (d, m, n) containing the B-spline basis functions evaluated at the points x.
    """
    B = @tullio out[n, d, m] := exp(-sum((x[n, d] - grid[d, m])^2) / (2*σ[1]^2))
    return B
end

BasisFcn = Dict(
    "spline" => B_batch,
    "RBF" => B_batch_RBF
)[method]

function coef2curve(x_eval, grid, coef; k::Int64, scale=1.0)
    """
    Compute the B-spline curves from the B-spline coefficients.

    Args:
        x_eval: A matrix of size (d, n) containing the points at which to evaluate the B-spline curves.
        grid: A matrix of size (d, m) containing the grid of knots.
        coef: A matrix of size (d, m, l, k) containing the B-spline coefficients.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (d, l, n) containing the B-spline curves evaluated at the points x_eval.
    """
    
    b_splines = BasisFcn(x_eval, grid; degree=k, σ=scale)
    y_eval = @tullio out[i, j, l] := b_splines[i, j, k] * coef[j, l, k]
    return y_eval
end

function curve2coef(x_eval, y_eval, grid; k::Int64, scale=1.0)
    """
    Convert B-spline curves to B-spline coefficients using least squares.

    Args:
        x_eval: A matrix of size (d, n) containing the points at which the B-spline curves were evaluated.
        y_eval: A matrix of size (d, l, n) containing the B-spline curves evaluated at the points x_eval.
        grid: A matrix of size (d, m) containing the grid of knots.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (d, m, l, k) containing the B-spline coefficients.
    """
    b_size = size(grid, 1)
    n_coeffs = size(grid, 2) - k - 1
    out_dim = size(y_eval, 3)
    B = BasisFcn(x_eval, grid; degree=k, σ=scale) 

    # Compute the B-spline coefficients using least squares with \ operator
    B = removeZero.(B)
    coef = @tullio out[j, q, p] := B[i, j, p] \ y_eval[i, j, q]
    
    return coef
end

end