module Spline

include("../utils.jl")

export extend_grid, B_batch, coef2curve, curve2coef

using .Utils: device
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
    h = (grid[:, end] .- grid[:, 1]) ./ (size(grid, 2) - 1)
    for i in 1:k_extend
        grid = hcat(grid[:, 1] .- h, grid)
        grid = hcat(grid, grid[:, end] .+ h)
    end
    return grid
end

function B_batch(x, grid; k::Int64, extend=true, eps=1e-4)
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
    if extend
        grid = extend_grid(grid, k)
    end

    d = size(grid, 1)
    n = size(x, 2)
    x = reshape(x, d, n, 1)
    
    B = zeros(d, size(grid, 2) - 1, n) |> device
    
    if k == 0
        B = (grid[:, 1:end-1] .<= x) .* (x .< grid[:, 2:end])
    else
        for i in 1:size(grid, 2) - 1
            B[:, i, :] .= (grid[:, i] .<= x) .* (x .< grid[:, i + 1])
        end
        for j in 1:k
            for i in 1:size(grid, 2) - j - 1
                B[:, i, :] .= (x .- grid[:, i]) ./ (grid[:, i + j] .- grid[:, i]) .* B[:, i, :] .+
                             (grid[:, i + j + 1] .- x) ./ (grid[:, i + j + 1] .- grid[:, i + 1]) .* B[:, i + 1, :]
            end
        end
    end

    B = replace(B, NaN=>eps)
    return B
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

function curve2coef(x_eval, y_eval, grid; k::Int64)
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
    b_size, in_dim = size(x_eval)
    out_dim = size(y_eval, 2)
    B = B_batch(x_eval, grid; k)
    coefs = @tullio coef[d, p] := y_eval[d, n] / B[d, p, n] 
    return coef
end

end