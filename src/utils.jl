module Utils

export device

using Flux, CUDA, KernelAbstractions, LinearAlgebra

const USE_GPU = CUDA.has_cuda() && parse(Bool, get(ENV, "GPU", "false"))

function device(x)
    return USE_GPU ? gpu(x) : x
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
    
    in_coord = (collect(0:in_dim-1) .+ 0.5) ./ in_dim
    out_coord = (collect(0:out_dim-1) .+ 0.5) ./ out_dim

    dist_mat = abs.(out_coord' .- in_coord)
    in_nearest = argmin(dist_mat, dims=1)
    in_connection = hcat(collect(1:in_dim), in_nearest')

    out_nearest = argmin(dist_mat, dims=2)
    out_connection = hcat(out_nearest, collect(1:out_dim))

    all_connection = vcat(in_connection, out_connection)
    mask = zeros(in_dim, out_dim)

    for i in 1:size(all_connection, 1)
        mask[all_connection[i, 1], all_connection[i, 2]] = 1.0
    end

    return mask
end

end