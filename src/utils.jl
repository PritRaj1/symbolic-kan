module Utils

# export device

using Flux, Tullio, LinearAlgebra, Statistics, GLM, DataFrames, Random
# using CUDA, KernelAbstractions

# const USE_GPU = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))

# function device(x)
#     return USE_GPU ? gpu(x) : x
# end


function removeNaN(x)
    return isnan(x) ? Float32(0.0) : x
end

function removeZero(x; ε=1e-3)
    return iszero(x) ? Float32(ε) : x
end

# Smooth sigmoid approximation of thresholding
smooth_transition1(x, y; steepness=5.0f0) = σ(steepness * (x - y))
smooth_transition2(x, y; steepness=5.0f0) = 1.0f0 - σ(steepness * (x - y))

function sparse_mask(in_dim, out_dim)
    """
    Create a sparse mask for the KAN layer.

    Args:
        in_dim: The number of input dimensions.
        out_dim: The number of output dimensions.

    Returns:
        A sparse mask of size (in_dim, out_dim) with 1s at the nearest connections.
    """

    in_coord = range(1, in_dim, step=1) |> collect
    out_coord = range(1, out_dim, step=1) |> collect
    in_coord = in_coord .* (1 / (2 * in_dim^2))
    out_coord = out_coord .* (1 / (2 * out_dim^2))
    
    dist_mat = abs.(out_coord' .- in_coord)
    in_nearest = argmin(dist_mat, dims=1)
    in_connection = hcat(collect(1:in_dim), in_nearest')

    out_nearest = argmin(dist_mat, dims=2)
    out_connection = hcat(out_nearest, collect(1:out_dim))


    all_connection = vcat(in_connection, out_connection)
    mask = zeros(in_dim, out_dim)

    for i in eachindex(all_connection[:, 1])
        mask[all_connection[i, 1], all_connection[i, 2]] = 1.0
    end

    return mask
end

function meshgrid(x, y)
    """
    Create a meshgrid from two 1D arrays.

    Args:
        x: 1D array.
        y: 1D array.

    Returns:
        x_grid: meshgrid of x.
        y_grid: meshgrid of y.
    """

    Nx, Ny = length(x), length(y)
    x_out, y_out = zeros(Ny, Nx), zeros(Ny, Nx)

    for i in 1:Nx
        for j in 1:Ny
            x_out[j, i] = x[i]
            y_out[j, i] = y[j]
        end
    end

    return x_out, y_out
end

function expand_apply(fcn, x, α, β; grid_number)
    x = reshape(x, length(x), 1, 1)
    α = reshape(α, 1, grid_number, grid_number)
    β = reshape(β, 1, grid_number, grid_number)
    eval = @tullio res[i, j, k] := fcn(α[1, j, k] * x[i, 1, 1] + β[1, j, k])
    return fcn.(eval)
end

end