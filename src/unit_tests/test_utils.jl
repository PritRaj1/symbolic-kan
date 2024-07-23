using Test, Random
using Plots; pythonplot()

include("../utils.jl")
using .Utils: fit_params, create_loaders

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

"""
num = 100
x = torch.linspace(-1,1,steps=num)
# noises = torch.normal(0,1,(num,)) * 0.02
y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 #+ noises
fit_params(x, y, torch.sin)
# r2 is 0.9999727010726929
# (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))"""

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

# test_param_fitting()
# test_sin_fitting()
test_loaders()