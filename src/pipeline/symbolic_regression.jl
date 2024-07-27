module SymbolicRegression

export fit_params, fix_symbolic!, unfix_symbolic!, unfix_symb_all!, suggest_symbolic!

using Flux, Tullio, LinearAlgebra, Statistics, GLM, DataFrames, Random, SymPy

include("../architecture/kan_model.jl")
include("../symbolic_lib.jl")
using .KolmogorovArnoldNets: set_mode!, lock_symbolic!
using .SymbolicLib: SYMBOLIC_LIB

function fit_params(x, y, fcn; α_range=(-10, 10), β_range=(-10, 10), grid_number=101, iterations=3, μ=1.0, verbose=true)
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
    squared_err = sum((y .- f_approx_best).^2)

    if verbose == true
        println("Best α: ", α_best)
        println("Best β: ", β_best)
        println("Best w: ", w_best)
        println("Best b: ", b_best)
        println("Best R2: ", R2_best)
        println("Squared Error: ", squared_err)
        squared_err <= 1.0 ? println("Good fit!") : println("Poor fit! Check symbolic function.")
    end

    return [α_best, β_best, w_best, b_best], R2_best
end

function fix_symbolic!(model, l, i, j, fcn_name; fit_params=true, α_range=(-10, 10), β_range=(-10, 10), grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=true)
    """
    Set the activation for element (l, i, j) to a fixed symbolic function.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        fcn_name: Name of the symbolic function.
        fit_params: Fit the parameters of the symbolic function.
        α_range: Range of the α parameter in fit.
        β_range: Range of the β parameter in fit.
        grid_number: Number of grid points in fit.
        iterations: Number of iterations in fit.
        μ: Step size in fit.
        random: Random setting.
        verbose: Print updates.

    Returns:
        R2 (or nothing): Coefficient of determination.
    """
    set_mode!(model, l, i, j, "s")
    
    if !fit_params
        R2 = lock_symbolic!(model.symbolic_fcns[l], i, j, fcn_name)
        return nothing
    else
        x = model.acts[l]
        y = model.post_acts[l][:, i, j]
        R2 = lock_symbolic!(model.symbolic_fcns[l], i, j, fcn_name; x=x, y=y, α_range=α_range, β_range=β_range, μ=μ, random=random, seed=seed, verbose=verbose)
        return R2
    end 
end

function unfix_symbolic!(model, l, i, j)
    """
    Unfix the symbolic function for element (l, i, j).

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
    """
    return set_mode!(model, l, i, j, "n")
end

function unfix_symb_all!(model)
    """
    Unfix all symbolic functions in the model.
    """
    for l in 1:model.depth
        for i in 1:model.widths[l]
            for j in 1:model.widths[l + 1]
                model = unfix_symbolic!(model, l, i, j)
            end
        end
    end
end

function suggest_symbolic!(model, l, i, j; α_range=(-10, 10), β_range=(-10, 10), lib=nothing, top_K=5, verbose=true)
    """
    Suggest potential symbolic functions for φ(l, i, j).

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        α_range: Range of the α parameter in fit.
        β_range: Range of the β parameter in fit.
        lib: Symbolic library.
        top_K: Number of top functions to suggest.
        verbose: Print updates.

    Returns:    
        - best_name: Name of the best symbolic function.
        - best_fcn: Best symbolic function.
        - best_R2: Coefficient of determination.
    """
    R2s = []
    symbolic_lib = isnothing(lib) ? SYMBOLIC_LIB : Dict{SYMBOLIC_LIB[k] for k in lib}
    
    for (name, fcn) in symbolic_lib
        R2 = fix_symbolic!(model, l, i, j, name; fit_params=true, α_range=α_range, β_range=β_range, grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=verbose)
        push!(R2s, R2)
    end

    unfix_symbolic!(model, l, i, j)
    sorted_R2s = sortperm(R2s, rev=true)
    top_K = min(top_K, length(sorted_R2s))
    top_R2s = sorted_R2s[1:top_K]

    if verbose
        println("Top ", top_K, " symbolic functions for φ(", l, ", ", i, ", ", j, "):")
        for i in 1:top_K
            println("Name: ", symbolic_lib[top_R2s[i]][1], " R2: ", R2s[top_R2s[i]])
        end
    end

    best_name = symbolic_lib[top_R2s[1]][1]
    best_fcn = symbolic_lib[top_R2s[1]][2]
    best_R2 = R2s[top_R2s[1]]

    return best_name, best_fcn, best_R2
end

function auto_symbolic!(model; α_range=(-10, 10), β_range=(-10, 10), lib=nothing, verbose=true)
    """
    Automatically replace all splines in the model with best fitting symbolic functions.
    
    Args:
        α_range: Range of the α parameter in fit.
        β_range: Range of the β parameter in fit.
        lib: Symbolic library.
        verbose: Print updates.
    """
    for l in eachindex(model.widths[1:end-1])
        for i in 1:model.widths[l]
            for j in 1:model.widths[l+1]
                if model.symbolic_fcns[l].mask[j, i] > 0.0
                    println("Skipping φ(", l, ", ", i, ", ", j, ") as it is already symbolic.")
                else
                    best_name, best_fcn, best_R2 = suggest_symbolic!(model, l, i, j; α_range=α_range, β_range=β_range, lib=lib, top_K=5, verbose=verbose)
                    fix_symbolic!(model, l, i, j, best_name; fit_params=true, α_range=α_range, β_range=β_range, grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=verbose)
                    if verbose
                        println("Suggested: ", best_name, " for φ(", l, ", ", i, ", ", j, ") with R2: ", best_R2)
                    end
                end
            end
        end
    end
end


function ex_round(ex1, floating_digit)
    ex2 = ex1
    for a in sympy.preorder_traversal(ex1)
        if isa(a, sympy.Float)
            ex2 = subs(ex2, a => round(a, digits=floating_digit))
        end
    end
    return ex2
end

function symbolic_formula(model, l, i, j; α_range=(-10, 10), β_range=(-10, 10), verbose=true)
    """
    Suggest the best symbolic function for φ(l, i, j).

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        α_range: Range of the α parameter in fit.
        β_range: Range of the β parameter in fit.
        verbose: Print updates.

    Returns:
        nothing, print out the best symbolic function.
    """
    symbolic_acts = []
    x = []
end

end

