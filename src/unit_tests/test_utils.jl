using Test, Random
using Plots; pythonplot()

include("../utils.jl")
include("../pipeline/utils.jl")
using .Utils: fit_params
using .PipelineUtils

# Test parameter fitting for symbolic reg
function test_param_fitting()
    num = 100
    x = range(-1, 1, length=num) |> collect
    noises = randn(num) .* 0.02
    y = 2 .* x .+ 1 .+ noises
    fcn = x -> x
    params, R2 = fit_params(x, y, fcn)

    @test R2 >= 0.9
    @test abs(params[1] - 2) < 0.01
    @test abs(params[2] - 1) < 0.01
    @test abs(params[3] - 1) < 0.01
    @test abs(params[4] - 0) < 0.01
end

function test_sin_fitting()
    num = 100
    x = range(-1, 1, length=num) |> collect
    Random.seed!(123)
    noises = randn(num) .* 0.02
    y = 5 .* sin.(3 .* x .+ 2) .+ 0.7 .+ noises
    fcn(x) = sin(x)
    params, R2 = fit_params(x, y, fcn)

    # Plot
    plot(x, y, label="data")
    plot!(x, fcn.(3 .* x .+ 2) .* 5 .+ 0.7, label="true")
    plot!(x, fcn.(params[1] .* x .+ params[2]) .* params[3] .+ params[4], label="fit")
    savefig("figures/test_sin_fitting.png")
end

function test_loaders()
    fcn = x -> x
    train_loader, test_loader = create_loaders(fcn)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 2))
    @test all(size(y |> permutedims) .== (32, 2))
    @test all(x .- y .== 0.0)

    train_loader, test_loader = create_loaders(fcn; N_var=3)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 3))
    @test all(size(y |> permutedims) .== (32, 3))
    @test all(x .- y .== 0.0)
end

function test_optimisers()
    optimiser, scheduler = create_optimiser("adam"; LR=0.01)
    @test scheduler(1, 0.01) ≈ 0.01
    optimiser, scheduler = create_optimiser("lbfgs"; schedule_LR=true, LR=0.01, step=10, decay=0.1, min_LR=0.001)
    @test scheduler(50, 0.01) ≈ 0.001
end

test_param_fitting()
test_sin_fitting()
test_loaders()
test_optimisers()