using Test, GLMakie, Random

include("../pipeline/symbolic_regression.jl")
include("../architecture/kan_model.jl")
include("../pipeline/utils.jl")
include("../pipeline/flux_trainer.jl")
include("../pipeline/optimisation.jl")
include("../pipeline/plot.jl")
include("../utils.jl")
using .KolmogorovArnoldNets
using .SymbolicRegression
using .PipelineUtils
using .FluxTrainer
using .Optimisation
using .Plotting
using .Utils: round_formula

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
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
    lines!(ax, x, y, label="data")
    lines!(ax, x, fcn.(3 .* x .+ 2) .* 5 .+ 0.7, label="true")
    lines!(ax, x, fcn.(params[1] .* x .+ params[2]) .* params[3] .+ params[4], label="fit")
    save("figures/test_sin_fitting.png", fig)
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

function test_lock_symb()
    layer = symbolic_kan_layer(3, 2)
    lock_symbolic!(layer, 3, 2, "sin")

    @test layer.fcns[2][3](2.4) ≈ sin(2.4)
    @test layer.fcn_names[2][3] == "sin"
    @test all(layer.affine[2, 3, :] .≈ [1.0, 0.0, 1.0, 0.0])

    Random.seed!(123)
    layer = symbolic_kan_layer(3, 2)
    num = 100
    x = range(-1, 1, length=num) |> collect
    noises = randn(num) .* 0.02
    y = 2 .* x .+ 1 .+ noises
    fcn = "x"
    R2 = lock_symbolic!(layer, 3, 2, fcn; x, y, random=true, seed=123)

    @test layer.fcns[2][3](2.4) ≈ 2.4
    @test layer.fcn_names[2][3] == "x"
    @test layer.affine[2, 3, 1] - 2 < 0.01
    @test layer.affine[2, 3, 2] - 1 < 0.01
    @test layer.affine[2, 3, 3] - 1 < 0.01
    @test layer.affine[2, 3, 4] - 0 < 0.01
    @test R2 >= 0.9
end

function test_suggestion()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    f = x -> exp(sin(π*x[1] + x[2]^2))
    train_loader, test_loader = create_loaders(f, N_var=2, x_range=(-1,1), N_train=100, N_test=100, batch_size=10, init_seed=1234)
    lr_scheduler = step_decay_scheduler(5, 0.8, 1e-5)
    opt = create_flux_opt(model, "adam"; LR=0.01, decay_scheduler=lr_scheduler)
    trainer = init_flux_trainer(model, train_loader, test_loader, opt; max_epochs=100, verbose=true)
    train!(trainer; λ=0.01)
    suggest_symbolic!(model, 1, 1, 1)
end

function test_auto()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    f = x -> exp(sin(π*x[1] + x[2]^2))
    train_loader, test_loader = create_loaders(f, N_var=2, x_range=(-1,1), N_train=2000, N_test=2000, batch_size=100, init_seed=1234)
    lr_scheduler = step_decay_scheduler(5, 0.98, 1e-2)
    opt = create_flux_opt(model, "adam"; LR=0.01, decay_scheduler=lr_scheduler)
    trainer = init_flux_trainer(model, train_loader, test_loader, opt; max_epochs=150, verbose=true, update_grid_bool=false)
    train!(trainer; λ=1, grid_update_num=5)
    model = prune(model)
    x = first(train_loader)[1] |> permutedims
    fwd!(model, x) 
    auto_symbolic!(model; lib=["exp","sin","x^2"])
    return model
end



function test_formula(model)
    formula, _ = symbolic_formula!(model)
    println(formula[1])

    formula = string(formula[1])
    return round_formula(formula)
end

function plot_symb(model, form)
    plot_kan!(model; mask=true, in_vars=["x1", "x2"], out_vars=[string(formula)], title="KAN")
end

# test_param_fitting()
# test_sin_fitting()
# test_lock()
# test_lock_symb()
# test_suggestion()
model = test_auto()
formula = test_formula(model)
plot_symb(model, formula)