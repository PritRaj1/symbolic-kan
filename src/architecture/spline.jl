ENV["GPU"] = "true"

include("../utils.jl")

using .Utils: device

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

function B_batch(x, grid; k::Int64, extend=true)
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
    
    return B
end

# Test
num_spline = 5
num_sample = 100
num_grid_interval = 10
k = 3
x = randn(num_spline, num_sample) |> device
grids = randn(num_spline, 11) |> device
u = B_batch(x, grids; k=k)
println(size(u))  # Expected output: [5, 14, 100]
