module Optimisation

export create_optim_opt, opt_get

using Lux, OptimizationOptimJL, LineSearches, Optimisers

## Optim optimiser ##
struct optim_opt
    type
    line_search
    m::Int
    init_α::Float32
end

function create_optim_opt(type="l-bfgs", line_search="strongwolfe"; m=10, c_1=1e-4, c_2=0.9, ρ=0.5, init_α=0.1)
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
    
    linesearch_map = Dict(
        "strongwolfe" => LineSearches.StrongWolfe(c_1=Float32(c_1), c_2=Float32(c_2), ρ=Float32(ρ)),
        "backtrack" => LineSearches.BackTracking(c_1=Float32(c_1), ρ_hi=Float32(ρ), ρ_lo=Float32(0.1), maxstep=Inf32),
        "hagerzhang" => LineSearches.HagerZhang(),
        "morethuente" => LineSearches.MoreThuente(),
        "static" => LineSearches.Static(),
    )
    
    line_search = linesearch_map[line_search]

    return optim_opt(type, line_search, m, init_α)
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
        return Optim.LBFGS(m=o.m, linesearch=o.line_search, alphaguess=Float32(1))
    elseif o.type == "neldermead"
        return Optim.NelderMead()
    else
        optimiser_map = Dict(
            "bfgs" => Optim.BFGS(linesearch=o.line_search, alphaguess=InitialHagerZhang(α0=o.init_α)),
            "cg" => Optim.ConjugateGradient(linesearch=o.line_search, alphaguess=InitialHagerZhang(α0=o.init_α)),
            "gd" => Optim.GradientDescent(linesearch=o.line_search, alphaguess=InitialHagerZhang(α0=o.init_α)),
            "newton" => Optim.Newton(linesearch=o.line_search, alphaguess=InitialHagerZhang(α0=o.init_α)),
            "interior-point" => Optim.IPNewton(linesearch=o.line_search),
        )
        return optimiser_map[o.type]
    end

end

end