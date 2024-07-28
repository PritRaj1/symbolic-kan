using Test
using Plots; pythonplot()

include("../pipeline/symbolic_regression.jl")
include("../architecture/kan_model.jl")
include("../pipeline/utils.jl")
include("../pipeline/flux_trainer.jl")
using .KolmogorovArnoldNets
using .SymbolicRegression
using .PipelineUtils
using .Trainer

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

function test_lock()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    fix_symbolic!(model, 1, 2, 4, "sin"; fit_params=false)
    mask1 = model.act_fcns[1].mask
    mask2 = model.symbolic_fcns[1].mask
    @test all(mask1[1, :] .== [1.0, 1.0, 1.0, 1.0, 1.0])
    @test all(mask2[:, 1] .== [1.0, 1.0, 1.0, 1.0, 1.0])
end

function test_suggestion()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    f = x -> exp(sin(π*x[1] + x[2]^2))
    train_loader, test_loader = create_loaders(x -> x[1] * x[2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, batch_size=10, init_seed=1234)
    lr_scheduler = step_decay_scheduler(5, 0.8, 1e-5)
    opt = create_opt(model, "adam"; LR=0.0001, decay_scheduler=lr_scheduler)
    trainer = init_flux_trainer(model, train_loader, test_loader, opt; max_epochs=100, verbose=true)
    train!(trainer)
    suggest_symbolic!(model, 1, 1, 1)
end

function test_auto()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    f = x -> exp(sin(π*x[1] + x[2]^2))
    train_loader, test_loader = create_loaders(x -> x[1] * x[2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, batch_size=10, init_seed=1234)
    lr_scheduler = step_decay_scheduler(5, 0.8, 1e-5)
    opt = create_opt(model, "adam"; LR=0.0001, decay_scheduler=lr_scheduler)
    trainer = init_flux_trainer(model, train_loader, test_loader, opt; max_epochs=100, verbose=true)
    train!(trainer)
    auto_symbolic!(model; lib=['exp','sin','x^2'])
end

# test_param_fitting()
# test_sin_fitting()
# test_lock()
test_suggestion()
test_auto()