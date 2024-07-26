module SymbolicRegression

export fix_symbolic!, unfix_symbolic!, unfix_symb_all!

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets: set_mode!, lock_symbolic!

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

end