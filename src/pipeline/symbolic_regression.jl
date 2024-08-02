module SymbolicRegression

export fit_params, set_affine, lock_symbolic, set_mode, fix_symbolic, unfix_symbolic, unfix_symb_all, suggest_symbolic, auto_symbolic, symbolic_formula

using Flux, Tullio, LinearAlgebra, Statistics, GLM, DataFrames, Random, SymPy, Accessors

include("../symbolic_lib.jl")
include("../architecture/symbolic_layer.jl")
include("../utils.jl")
using .SymbolicLib: SYMBOLIC_LIB
using .Utils: meshgrid, expand_apply

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
    y = Float64.(y) # GLM needs Vector{Float64}
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

function set_affine(ps, j, i; a1=1.0, a2=0.0, a3=1.0, a4=0.0)
    """
    Set affine parameters for symbolic dense layer.
    
    Args:
    - l: symbolic dense layer.
    - j: index of output neuron.
    - i: index of input neuron.
    - a1: param1.
    - a2: param2.
    - a3: param3.
    - a4: param4.
    """
    @reset ps[j, i, 1] = a1
    @reset ps[j, i, 2] = a2
    @reset ps[j, i, 3] = a3
    @reset ps[j, i, 4] = a4

    return ps
end

function lock_symbolic(l, ps, i, j, fun_name; x=nothing, y=nothing, random=false, seed=nothing, α_range=(-10, 10), β_range=(-10, 10), μ=1.0, verbose=true)
    """
    Fix a symbolic function for a particular input-output pair, 
    
    i.e. the univariate fcn used for a particular evaluation of the Kolmogorov-Arnold theorem.

    Args:
    - l: symbolic dense layer.
    - i: index of input neuron.
    - j: index of output neuron.
    - fun_name: name of symbolic function to lock.
    - x: 1D array of preactivations
    - y: 1D array of postactivations
    - random: whether to randomly initialise function parameters.
    - α_range: sweep range for α parameter.
    - β: sweep range for β parameter.
    - verbose: whether to print updates.

    Returns:
    - R2: coefficient of determination.
    """
    # fun_name is a name of a symbolic function
    if fun_name isa String
        fcn = SYMBOLIC_LIB[fun_name][1]
        fcn_sympy = SYMBOLIC_LIB[fun_name][2]
        fcn_avoid_singular = SYMBOLIC_LIB[fun_name][3]

        @reset l.fcn_sympys[j][i] = fcn_sympy
        @reset l.fcn_names[j][i] = fun_name

        # If x and y are not provided, just set the function
        if isnothing(x) || isnothing(y)
            @reset l.fcns[j][i] = fcn
            @reset l.fcns_avoid_singular[j][i] = fcn_avoid_singular

            # Set affine parameters either to random values or to default values
            if !random
                ps = set_affine(ps, j, i)
            else
                Random.seed!(seed)
                params = rand(4) .* 2 .- 1
                ps = set_affine(ps, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
            end

        # If x and y are provided, fit the function
        else
            params, R2 = fit_params(x, y, fcn; α_range=α_range, β_range=β_range, μ=μ, verbose=false)
            @reset l.fcns[j][i] = fcn
            @reset l.fcns_avoid_singular[j][i] = fcn_avoid_singular
            ps = set_affine(ps, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
            return R2, l, ps
        end

    # fun_name is a symbolic function
    else
        @reset l.fcns[j][i] = fun_name
        @reset l.fcn_sympys[j][i] = fun_name
        @reset l.fcns_avoid_singular[j][i] = fun_name
        @reset l.fcn_names[j][i] = "anonymous"

        # Set affine parameters either to random values or to default values
        if !random
            ps = set_affine(ps, j, i)
        else
            Random.seed!(seed)
            params = rand(4) .* 2 .- 1
            ps = set_affine(ps, j, i; a1=params[1], a2=params[2], a3=params[3], a4=params[4])
        end
    end

    return nothing, l, ps
end

function set_mode(st, l, i, j, mode; mask_n=nothing)
    """
    Set neuron (l, i, j) to mode.

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
        mode: 'n' (numeric) or 's' (symbolic) or 'ns' (combined)
        mask_n: Magnitude of the mask for numeric neurons.
    """

    if mode == "s"
        mask_n = 0.0f0
        mask_s = 1.0f0
    elseif mode == "n"
        mask_n = 1.0f0
        mask_s = 0.0f0
    elseif mode == "sn" || mode == "ns"
        if isnothing(mask_n)
            mask_n = 1.0f0
        else
            mask_n = mask_n
        end
        mask_s = 1.0f0
    else
        mask_n = 0.0f0
        mask_s = 0.0f0
    end

    @reset st.act_fcns_st[l].mask[i, j] = mask_n
    @reset st.symbolic_fcns_st[l].mask[j, i] = mask_s

    return st
end

function fix_symbolic(model, ps, st, l, i, j, fcn_name; fit_params=true, α_range=(-10, 10), β_range=(-10, 10), grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=true)
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
    st = set_mode(st, l, i, j, "s")
    
    if !fit_params
        R2, new_l, new_ps = lock_symbolic(model.symbolic_fcns[l], ps.symbolic_fcns_ps[Symbol("layer_$l")], i, j, fcn_name)
        @reset model.symbolic_fcns[l] = new_l
        @reset ps.symbolic_fcns_ps[Symbol("layer_$l")] = new_ps
        return nothing, model, ps, st
    else
        x = st.acts[l][:, i]
        y = st.post_acts[l][:, j, i]
        R2, new_l, new_ps = lock_symbolic(model.symbolic_fcns[l], ps.symbolic_fcns_ps[Symbol("layer_$l")], i, j, fcn_name; x=x, y=y, α_range=α_range, β_range=β_range, μ=μ, random=random, seed=seed, verbose=verbose)
        @reset model.symbolic_fcns[l] = new_l
        @reset ps.symbolic_fcns_ps[Symbol("layer_$l")] = new_ps
        return R2, model, ps, st
    end 
end

function unfix_symbolic(st, l, i, j)
    """
    Unfix the symbolic function for element (l, i, j).

    Args:
        l: Layer index.
        i: Neuron input index.
        j: Neuron output index.
    """
    return set_mode(st, l, i, j, "n")
end

function unfix_symb_all(model, st)
    """
    Unfix all symbolic functions in the model.
    """
    for l in 1:model.depth
        for i in 1:model.widths[l]
            for j in 1:model.widths[l + 1]
                st = unfix_symbolic!(model, l, i, j)
            end
        end
    end
    return st
end

function suggest_symbolic(model, ps, st, l, i, j; α_range=(-10, 10), β_range=(-10, 10), lib=nothing, top_K=5, verbose=true)
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
    if isnothing(lib)
        symbolic_lib = SYMBOLIC_LIB
    else
        symbolic_lib = Dict()
        for name in lib
            symbolic_lib[name] = SYMBOLIC_LIB[name]
        end
    end
    
    for (name, fcn) in symbolic_lib
        R2, model, ps, st = fix_symbolic(model, ps, st, l, i, j, name; fit_params=true, α_range=α_range, β_range=β_range, grid_number=101, iterations=3, μ=1.0, random=false, seed=nothing, verbose=verbose)
        push!(R2s, R2)
    end

    st = unfix_symbolic(st, l, i, j)
    sorted_R2s = sortperm(R2s, rev=true)
    top_K = min(top_K, length(sorted_R2s))
    top_R2s = sorted_R2s[1:top_K]

    if verbose
        println("Top ", top_K, " symbolic functions for φ(", l, ", ", i, ", ", j, "):")
        for i in 1:top_K
            println("Name: ", collect(symbolic_lib)[top_R2s[i]][1], " R2: ", R2s[top_R2s[i]])
        end
    end

    best_name = collect(symbolic_lib)[top_R2s[1]][1]
    best_fcn = collect(symbolic_lib)[top_R2s[1]][2]
    best_R2 = R2s[top_R2s[1]]

    return model, ps, st, best_name, best_fcn, best_R2
end

function auto_symbolic(model, ps, st; α_range=(-10, 10), β_range=(-10, 10), lib=nothing, verbose=true)
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
                if st.symbolic_fcns_st[l].mask[j, i] > 0.0
                    println("Skipping φ(", l, ", ", i, ", ", j, ") as it is already symbolic.")
                else
                    model, ps, st, best_name, best_fcn, best_R2 = suggest_symbolic(model, ps, st, l, i, j; α_range=α_range, β_range=β_range, lib=lib, top_K=5, verbose=verbose)
                    _, model, ps, st = fix_symbolic(model, ps, st, l, i, j, best_name; fit_params=true, α_range=α_range, β_range=β_range, grid_number=201, iterations=10, μ=1.0, random=false, seed=nothing, verbose=verbose)
                    if verbose
                        println("Suggested: ", best_name, " for φ(", l, ", ", i, ", ", j, ") with R2: ", best_R2)
                    end
                end
            end
        end
    end
    return model, ps, st
end

function symbolic_formula(model, ps, st; var=nothing, normaliser=nothing, output_normaliser=nothing, simplify=false)
    """
    Convert the activations of a model to symbolic formulas.

    Args:
    - model: KAN model.
    - floating_digit: Number of floating digits.
    - var: List of variable names.
    - normaliser: Tuple of mean and std for normalisation.
    - output_normaliser: Tuple of mean and std for output normalisation.
    - simplify: Whether to simplify the symbolic formulas.

    Returns:
    - symbolic_acts: List of symbolic activations.
    - x0: List of symbolic variables.
    """
    symbolic_acts = []
    x = []

    # Create symbolic variables
    if isnothing(var)
        for i in 1:model.widths[1]+1
            push!(x, sympy.Symbol("x$i"))
        end
    else
        x = [sympy.Symbol(var_) for var_ in var]
    end

    x0 = x

    if !isnothing(normaliser)
        mean, std = normaliser
        x = [(x[i] - mean[i]) / std[i] for i in eachindex(x)]
    end

    push!(symbolic_acts, x)

    # Convert activations to symbolic formulas
    for l in eachindex(model.widths[1:end-1])
        y = []
        for j in 1:model.widths[l+1]
            yj = 0.0
            for i in 1:model.widths[l]
                a, b, c, d = ps.symbolic_fcns_ps[Symbol("layer_$l")][j, i, :]
                
                try 
                    sympy_fcn = model.symbolic_fcns[l].fcns[j][i]
                    yj += c * sympy_fcn(a * x[i] + b) + d
                catch
                    println("Make sure all activations need to be converted to symbolic formulas first!")
                end

            end

            if simplify
                push!(y, sympy.simplify(yj + model.biases[l][1, j]))
            else
                push!(y, yj + ps.biases[Symbol("layer_$l")][1, j])
            end

        end
        x = y
        push!(symbolic_acts, x)

        output_lyr = symbolic_acts[end]
        if !isnothing(output_normaliser)
            mean, std = output_normaliser
            output_lyr = [(output_lyr[i] * std[i]) + mean[i] for i in eachindex(output_lyr)]
            symbolic_acts[end] = output_lyr
        end

        new_symbolic_acts = [[symbolic_acts[l][i] for i in eachindex(symbolic_acts[l])] for l in eachindex(symbolic_acts)]

        st = (
        act_fcns_st=st.act_fcns_st,
        symbolic_fcns_st=st.symbolic_fcns_st,
        acts = st.acts,
        pre_acts = st.pre_acts,
        post_acts = st.post_acts,
        post_splines = st.post_splines,
        act_scale = st.act_scale,
        mask = st.mask,
        symbolic_acts = new_symbolic_acts
        )

        return [symbolic_acts[end][i] for i in eachindex(symbolic_acts[end])], x0, st
    
    end
end

end

