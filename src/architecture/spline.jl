module Spline

export extend_grid, B_batch, coef2curve, curve2coef

using Flux, Tullio, LinearAlgebra
# using CUDA, KernelAbstractions

include("../utils.jl")
using .Utils: removeNaN, removeZero, smooth_transition1, smooth_transition2

method = get(ENV, "METHOD", "spline") # "spline" or "RBF"; RBF not properly implemented yet

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
    
    # B-spline basis functions of degree 0 are piecewise constant functions: B = 1 if x in [grid[p], grid[p+1]) else 0
    if degree == 0
        # Expand for broadcasting
        x_eval = repeat(reshape(x, size(x)..., 1), 1, 1, size(grid, 2) - 1)
        grid_eval = repeat(reshape(grid, 1, size(grid)...), size(x, 1), 1, 1)

        grid_1 = grid_eval[:, :, 1:end-1] # grid[p]
        grid_2 = grid_eval[:, :, 2:end] # grid[p+1]
    
        # Apply the smooth thresholding functions
        term1 = ifelse.(x_eval .>= grid_1, Float32(1), Float32(0))
        term2 = ifelse.(x_eval .< grid_2, Float32(1), Float32(0))

        B = @tullio res[d, p, n] := term1[d, p, n] * term2[d, p, n]
    
    else
        # Compute the B-spline basis functions of degree k
        k = degree
        B = B_batch(x, grid; degree=k-1)
        x = reshape(x, size(x)..., 1) 
        grid = reshape(grid, 1, size(grid)...) 

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

# function B_batch_RBF(x, grid; degree=nothing, σ=1.0)
#     """
#     Compute the B-spline basis functions for a batch of points x and a grid of knots using the RBF kernel.

#     Args:
#         x: A matrix of size (d, n) containing the points at which to evaluate the B-spline basis functions.
#         grid: A matrix of size (d, m) containing the grid of knots.
#         sigma: The bandwidth of the RBF kernel.

#     Returns:
#         A matrix of size (d, m, n) containing the B-spline basis functions evaluated at the points x.
#     """
#     B = @tullio out[n, d, m] := exp(-sum((x[n, d] - grid[d, m])^2) / (2*σ[1]^2))
#     return B

# end

BasisFcn = Dict(
    "spline" => B_batch,
    # "RBF" => B_batch_RBF
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
    y_eval = @tullio out[i, j, l] := b_splines[i, j, p] * coef[j, l, p]
    return y_eval
end

function curve2coef(x_eval, y_eval, grid; k::Int64, scale=1.0, ε=1e-4)
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
    b_size = size(x_eval, 1)
    in_dim = size(x_eval, 2)
    n_coeffs = size(grid, 2) - k - 1
    out_dim = size(y_eval, 3)
    B = BasisFcn(x_eval, grid; degree=k, σ=scale) 
    B = permutedims(B, [2, 1, 3])
    B = reshape(B, in_dim, 1, b_size, n_coeffs)
    B = repeat(B, 1, out_dim, 1, 1)

    y_eval = permutedims(y_eval, [2, 3, 1]) 

    # Get BtB and Bty
    Bt = permutedims(B, [1, 2, 4, 3])
    
    BtB = @tullio out[i, j, p, p] := Bt[i, j, p, n] * B[i, j, n, p]
    n1, n2, n, _ = size(BtB)
    eye = Matrix{Float32}(I, n, n) .* ε
    eye = reshape(eye, 1, 1, n, n)
    eye = repeat(eye, n1, n2, 1, 1)
    BtB = BtB .+ eye

    Bty = @tullio out[i, j, p] := Bt[i, j, p, n] * y_eval[i, j, n]
    
    # x = (BtB)^-1 * Bty
    coef = @tullio out[i, j, p] := inv(BtB[i, j, p, p]) * Bty[i, j, p]

    return coef

end
end