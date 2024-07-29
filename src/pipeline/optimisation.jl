module Optimisation

export step_decay_scheduler, create_flux_opt, create_optim_opt, opt_get

using Flux, Optim, LineSearches, Optimisers

### Step LR scheduler ### 
struct decay_scheduler
    step::Int
    decay::Float64
    min_LR::Float64
end

function step_decay_scheduler(step, decay, min_LR)
    return decay_scheduler(step, decay, min_LR)
end

function (s::decay_scheduler)(epoch, LR)
    return max(LR * s.decay^(epoch // s.step), s.min_LR)
end

### Flux optimiser ###
flux_map = Dict(
    "adam" => Optimisers.Adam,
    "sgd" => Optimisers.Descent
)

mutable struct flux_opt
    type
    opt_state
    LR_scheduler
    LR::Float32
end

function create_flux_opt(model, type="adam"; LR=0.01, decay_scheduler=nothing)
    """
    Create optimiser.

    Args:
    - type: optimiser to use.
    - schedule_LR: whether to schedule learning rate.
    - LR: learning rate.
    - step: step size for LR scheduler.
    - decay: decay rate for LR scheduler.
    - min_LR: minimum LR for LR scheduler.

    Returns:
    - optimiser: optimiser.
    """
    
    if !isnothing(decay_scheduler)
        schedule_fcn = decay_scheduler
    else
        schedule_fcn = (epoch, LR) -> LR
    end

    opt = flux_map[type](LR)
    opt_state = Optimisers.setup(opt, model)

    return flux_opt(type, opt_state, schedule_fcn, LR)
end

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

function create_optim_opt(model, type="l-bfgs", line_search="strongwolfe"; m=10, c_1=1e-4, c_2=0.9, ρ=0.5, ϵ=1e-8)
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
    else
        return Optim.NelderMead()
    end

end

end