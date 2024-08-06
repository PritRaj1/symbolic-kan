module Utils

# export device

using Lux, Tullio, LinearAlgebra, Statistics, GLM, DataFrames, Random
using CUDA, LuxCUDA, KernelAbstractions

const pu = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false")) ? gpu_device() : cpu_device()

function device(x)
    return pu(x)
end

function removeNaN(x)
    NaNs = @tullio res[i, j, k] := isnan(x[i, j, k])
    x = ifelse.(NaNs, Float32(0), x)
    return device(x)
end

function removeZero(x; ε=1e-3)
    return ifelse.(abs.(x) .< ε, Float32(ε), x)
end

# Rounds string formula to a certain number of digits.
function round_formula(formula; digits=3)
    return replace(formula, r"(\d+\.\d+)" => s -> string(round(parse(Float64, s), digits=digits)))
end

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

    return Float32.(mask)
end

function expand_apply(fcn, x, α, β; grid_number)
    """
    Creates meshgrids for α and β and applies the function fcn to the meshgrids.
    """
    α = ones(grid_number)' .* α
    β = β' .* ones(grid_number)

    eval = @tullio res[i, j, k] := fcn(α[j, k] * x[i] + β[j, k])
    return eval, α, β
end

end