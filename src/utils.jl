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
    x = ifelse.(NaNs, 0f0, x)
    return device(x)
end

function removeZero(x; ε=1f-3)
    return ifelse.(abs.(x) .< ε, ε, x)
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

    in_coord = 1:in_dim |> collect
    out_coord = 1:out_dim |> collect
    in_coord = in_coord .* (1 / (2 * in_dim^2)) |> Float32
    out_coord = out_coord .* (1 / (2 * out_dim^2)) |> Float32
    
    dist_mat = @tullio res[j, i] := abs(out_coord[i] - in_coord[j])
    in_nearest = [argmin(mat)[1] for mat in eachrow(dist_mat)] |> collect
    in_connection = hcat(collect(1:in_dim), in_nearest)

    out_nearest = [argmin(mat)[1] for mat in eachcol(dist_mat)] |> collect
    out_connection = hcat(out_nearest, collect(1:out_dim))

    all_connection = vcat(in_connection, out_connection)
    mask = zeros(Float32, in_dim, out_dim)

    for i in eachindex(all_connection[:, 1])
        mask[all_connection[i, 1], all_connection[i, 2]] = 1f0
    end

    return mask
end

function expand_apply(fcn, x, α, β; grid_number)
    """
    Creates meshgrids for α and β and applies the function fcn to the meshgrids.
    """
    α = ones(Float32, grid_number)' .* α
    β = β' .* ones(Float32, grid_number)

    eval = @tullio res[i, j, k] := fcn(α[j, k] * x[i] + β[j, k])
    return eval, α, β
end

end