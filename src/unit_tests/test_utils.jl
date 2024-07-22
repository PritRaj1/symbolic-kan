using Test

include("../utils.jl")
using .Utils: fit_params

num = 100
x = range(-1, 1, length=num) |> collect
noises = randn(num) .* 0.02
y = 2 .* x .+ 1 .+ noises
fcn = x -> x
params, R2 = fit_params(x, y, fcn)

@test R2 >= 0.9
@test abs(params[1] - 2) < 0.01
@test abs(params[2] - 1) < 0.01
@test abs(params[1] - 1) < 0.01
@test abs(params[4] - 0) < 0.01