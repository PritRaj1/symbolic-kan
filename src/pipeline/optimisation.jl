module Optimisation

export create_optim_opt, opt_get

using Lux, OptimizationOptimJL, LineSearches, Optimisers

## Optim optimiser ##
linesearch_map = Dict(
    "backtrack" => LineSearches.BackTracking(),
    "hagerzhang" => LineSearches.HagerZhang(),
    "morethuente" => LineSearches.MoreThuente(),
)

struct optim_opt
    type
    line_search
    m::Int
end

function create_optim_opt(type="l-bfgs", line_search="strongwolfe"; m=10, c_1=1e-4, c_2=0.9, ρ=0.5)
    """
    Create optimiser.

    Args:
    - type: optimiser to use.
    - line_search: line search to use.
    - m: memory size for L-BFGS.
    - c_1: Armijo condition constant.
    - c_2: Wolfe condition constant.
    - ρ: step size shrinkage factor.
    - ϵ: tolerance for termination.

    Returns:
    - optimiser: optimiser.
    """
    
    if line_search == "strongwolfe"
        line_search = LineSearches.StrongWolfe(c_1=c_1, c_2=c_2, ρ=ρ)
    else
        line_search = linesearch_map[line_search]
    end

    return optim_opt(type, line_search, m)
end

function opt_get(o)
    """
    Get optimiser.

    Args:
    - o: Optim optimiser object.

    Returns:
    - optimiser: optimiser.
    """

    if o.type == "l-bfgs" 
        return Optim.LBFGS(m=o.m, linesearch=o.line_search)
    elseif o.type == "bfgs"
        return Optim.BFGS(linesearch=o.line_search)
    elseif o.type == "cg"
        return Optim.ConjugateGradient(linesearch=o.line_search)
    elseif o.type == "gd"
        return Optim.GradientDescent(linesearch=o.line_search)
    else
        return Optim.NelderMead()
    end

end

end