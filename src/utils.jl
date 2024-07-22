module Utils

export device

using Flux, CUDA, KernelAbstractions, Tullio, LinearAlgebra, Statistics, GLM, DataFrames

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
    eval = @tullio res[i, j, k] := fcn(α[1, j, k] .* x[i, 1, 1] .+ β[1, j, k])
    return fcn.(eval)
end

function fit_params(x, y, fcn; α_range=(-10, 10), β_range=(-10, 10), grid_number=100, iterations=6, μ=3.0, verbose=true)
    """
    Optimises the parameters of a symbolic function to minismise l2-norm error (or maximise R2).
        
    i.e α, β, w, b = argmin ||y - (c*f(αx + β) + b) ||^2

    R2 = 1 - RSS / TSS = 1 - (sum(y - c*f(αx + β))^2 / sum((y - mean(y))^2))
    RSS = residual sum of squares, TSS = total sum of squares.

    Args:
    - x: 1D array of preactivations.
    - y: 1D array of target postactivations.
    - fcn: symbolic function.
    - α_range: sweep range for α parameter.
    - β: sweep range for β parameter.
    - grid_number: number of grid points for α, β.
    - iteration: number of iterations.
    - μ: step size for α, β.
    - verbose: whether to print updates.

    Returns:
    - α_best: best α parameter.
    - β_best: best β parameter.
    - w_best: best w parameter.
    - b_best: best b parameter.
    - R2: coefficient of determination.
    """

    α_best, β_best = nothing, nothing
    R2_best = nothing

    for _ in 1:iterations
        # Create search grids
        α_ = range(α_range[1], α_range[2], length=grid_number) |> collect
        β_ = range(β_range[1], β_range[2], length=grid_number) |> collect
        α_grid, β_grid = meshgrid(α_, β_)

        # Precompute f(αx + β) for each at all grid points
        y_approx = expand_apply(fcn, x, α_grid, β_grid; grid_number=grid_number)

        # Compute R2 for all grid points := 1 - (sum((y - f(αx + β)^2) / sum((y - mean(y))^2))
        RSS = @tullio res[i, j, k] := (y[i] - y_approx[i, j, k]) ^ 2
        RSS = sum(RSS, dims=1)[1, :, :]
        TSS = sum((y .- mean(y)).^2)
        R2 = @tullio res[j, k] := 1 - (RSS[j, k] / TSS)
        replace!(R2, NaN => 0.0)

        # Choose best α, β by maximising coefficient of determination
        best_id = argmax(R2)
        α_best = α_grid[best_id]
        β_best = β_grid[best_id]
        R2_best = R2[best_id]

        # Update α, β range for next iteration
        α_range = (α_best - μ*(α_range[2] - α_range[1]) / grid_number, α_best + μ*(α_range[2] - α_range[1]) / grid_number)
        β_range = (β_best - μ*(β_range[2] - β_range[1]) / grid_number, β_best + μ*(β_range[2] - β_range[1]) / grid_number)
    
    end

    # Linear regression to find w, b
    y_approx = fcn.(α_best .* x .+ β_best)
    df = DataFrame(X=y_approx, Y=y)
    model = lm(@formula(Y ~ X), df)
    b_best, w_best = coef(model)
    f_approx_best = y_approx .* w_best .+ b_best
    l2 = sum((y .- f_approx_best).^2)

    if verbose == true
        println("Best α: ", α_best)
        println("Best β: ", β_best)
        println("Best w: ", w_best)
        println("Best b: ", b_best)
        println("Best R2: ", R2_best)
        println("MSE: ", l2)
        R2_best >= 0.9 ? println("Good fit!") : println("Poor fit! Check symbolic function.")
    end

    return [α_best, β_best, w_best, b_best], R2_best
end

end