module Optimisation

export create_optim_opt, opt_get

using Lux, OptimizationOptimJL, LineSearches

## Optim optimiser ##
struct optim_opt
    type
    line_search
    m::Int
    init_α::Float32
    γ::Float32
end

function create_optim_opt(type="l-bfgs", line_search="strongwolfe"; m=10, c_1=1e-4, c_2=0.9, ρ=0.5, init_α=0.1, decay=0.9)
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
        "strongwolfe" => LineSearches.StrongWolfe{Float32}(c_1=Float32(c_1), c_2=Float32(c_2), ρ=Float32(ρ)),
        "backtrack" => LineSearches.BackTracking{Float32}(c_1=Float32(c_1), ρ_hi=Float32(ρ), ρ_lo=Float32(0.1), maxstep=Inf32),
        "hagerzhang" => LineSearches.HagerZhang{Float32}(),
        "morethuente" => LineSearches.MoreThuente{Float32}(f_tol=0f0, gtol=0f0, x_tol=0f0),
        # "static" => LineSearches.Static{Float32}(),
    )

    fcn = linesearch_map[line_search]
    line_fcn = (a...) -> fcn(a...) # Needed or else: ERROR: LoadError: TypeError: in keyword argument linesearch, expected Function, got a value of type LineSearches.StrongWolfe{Float32}
    return optim_opt(type, line_fcn, m, init_α, decay)
end

function opt_get(o; α=nothing)
    """
    Get optimiser.

    Args:
    - o: Optim optimiser object.

    Returns:
    - optimiser: optimiser.
    """

    init_α = isnothing(α) ? o.init_α : α
    
    optimiser_map = Dict(
        "bfgs" => BFGS(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=init_α), linesearch=o.line_search),
        "l-bfgs" => LBFGS(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=init_α), m=o.m, linesearch=o.line_search),
        "cg" => ConjugateGradient(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=init_α), linesearch=o.line_search),
        "gd" => GradientDescent(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=init_α), linesearch=o.line_search),
        "newton" => Newton(alphaguess=LineSearches.InitialHagerZhang{Float32}(α0=init_α), linesearch=o.line_search),
        "interior-point" => IPNewton(linesearch=o.line_search),
        "neldermead" => NelderMead(),
        "adam" => Adam(alpha=init_α, beta_mean=9f-1, beta_var=999f-3, epsilon=1f-8),
    )

    return optimiser_map[o.type]
end

end