module Spline

export extend_grid, B_batch, coef2curve, curve2coef

using CUDA, KernelAbstractions
using Tullio

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

function B_batch(x, grid; k::Int64, eps=1e-6)
    """
    Compute the B-spline basis functions for a batch of points x and a grid of knots.

    Args:
        x: A matrix of size (d, n) where d is the number of splines and n is the number of samples.
        grid: A matrix of size (d, m) where d is the number of splines and m is the number of evaluation points.
        k: The degree of the B-spline basis functions.
        extend: If true, extend the grid of knots to include the boundary knots.
    
    Returns:
        A matrix of size (d, p, n) containing the 'p' B-spline basis coefficients evaluated at the points x.
    """

    d, m = size(grid)
    n = size(x, 2)
    x = reshape(x, size(x)..., 1)
    grid = reshape(grid, 1, size(grid)...)

    if k == 0
        # B-spline basis functions of degree 0 are piecewise constant functions: B = 1 if x in [grid[p], grid[p+1]) else 0
        grid_1 = grid[:, :, 1:end-1] # grid[p]
        grid_2 = grid[:, :, 2:end] # grid[p+1]
        out = @tullio res[i, j, k] := (x[i, j, 1] >= grid_1[1, j, k]) * (x[i, j, 1] < grid_2[1, j, k])
    else
        # Compute the B-spline basis functions of degree 0:
        B_deg0 =  B_batch(x[:, :, 1], grid[1, :, :]; k=k-1) # Recurse through higher degree B-splines

        # Compute the B-spline basis functions of degree k:
        numer1 = x .- grid[:, :, 1:(end - k - 1)]
        denom1 = grid[:, :, (k + 1):end-1] .- grid[:, :, 1:(end - k - 1)]
        numer2 = grid[:, :, (k + 2):end] .- x
        denom2 = grid[:, :, (k + 2):end] .- grid[:, :, 2:(end - k)]
        out = numer1 ./ denom1 .* B_deg0[:, :, 1:end - 1] .+ numer2 ./ denom2 .* B_deg0[:, :, 2:end]
    end

    replace!(out, NaN=>eps)
    return out
end

function coef2curve(x_eval, grid, coef; k::Int64)
    """
    Compute the B-spline curves from the B-spline coefficients.

    Args:
        x_eval: A matrix of size (d, n) where d is the number of splines and n is the number of samples.
        grid: A matrix of size (d, m) where d is the number of splines and m is the number of grid points.
        coef: A matrix of size (d, p) where d is the number of splines and p is the number of coefficients.
        k: The degree of the B-spline basis functions.

    Returns:
        A matrix of size (d, n) containing the B-spline curves evaluated at the points x_eval.
    """
    b_splines = B_batch(x_eval, grid; k)
    y_eval = @tullio out[i, j] := b_splines[i, k, j] * coef[i, k] 
    return y_eval
end

function curve2coef(x_eval, y_eval, grid; k::Int64, eps=1e-6)
    """
    Convert B-spline curves to B-spline coefficients using least squares.

    Args:
        x_eval: A matrix of size (d, n) where d is the number of splines and n is the number of samples.
        y_eval: A matrix of size (d, n) where d is the number of splines and n is the number of samples.
        grid: A matrix of size (d, m) where d is the number of splines and m is the number of grid points.
        k: The piecewise polynomial order of splines.

    Returns:
        A matrix of size (d, p) containing the B-spline coefficients.
    """
    n_coeffs = size(grid, 2) - k - 1
    out_dim = size(y_eval, 3)
    B = B_batch(x_eval, grid; k) 
    B = permutedims(B, [2, 1, 3])
    B = reshape(B, size(B, 1), 1, size(B, 2), size(B, 3))
    B = repeat(B, 1, out_dim, 1, 1) 
    y_eval = permutedims(y_eval, [2, 3, 1])
    y_eval = reshape(y_eval, size(y_eval)..., 1) 

    # Least squares solution
    denom = replace(sum(B .* B, dims=3), 0.0=>eps)
    coef = sum(B .* y_eval, dims=3) ./ denom

    return coef[:, :, 1, :]
end

end