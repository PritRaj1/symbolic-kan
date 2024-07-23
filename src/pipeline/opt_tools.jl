module OptTools

using LineSearches, Optim

### StrongWolfe ###
struct SW
    c1::Float64
    c2::Float64
    ρ::Float64
end

function strong_wolfe(c1=1e-4, c2=0.9, ρ=2.0)
    return SW(c1, c2, ρ)
end

function (sw::SW)()
    return LineSearches.StrongWolfe(; c_1=sw.c1, c_2sw.c2, ρ=sw.ρ)
end

### HagerZhang ###
struct HW
end

function hager_zhang(c1=1e-4, c2=0.9, ρ=2.0)
    return HW()
end

function (hw::HW)()
    return LineSearches.HagerZhang()
end

### MoreThuente ###
struct MT
end

function more_thuente(c1=1e-4, c2=0.9, ρ=2.0)
    return MT()
end

function (mt::MT)()
    return LineSearches.MoreThuente()
end

### Static ###
struct Static
end

function static(c1=1e-4, c2=0.9, ρ=2.0)
    return Static()
end
function (st::Static)()
    return LineSearches.Static()
end

line_search_map = Dict(
    "strong_wolfe" => strong_wolfe,
    "hager_zhang" => hager_zhang,
    "more_thuente" => more_thuente,
    "static" => static
)

struct LBFGS
    line_search::Function
    m::Int
end

function lbfgs(line_search, history=100)
    return LBFGS(line_search, history)    
end

function (lb::LBFGS)(LR)
    return Optim.LBFGS(;m=lb.m, linesearch=lb.line_search(), P=LR)
end

struct GD
    line_search::Function
end

function gradient_descent(line_search, history=nothing)
    return GD(line_search)
end

function (gd::GD)(LR)
    return Optim.GradientDescent(; linesearch=gd.line_search(), P=LR)
end

optimiser_map = Dict(
    "lbfgs" => lbfgs,
    "gd" => gradient_descent
)

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

end
