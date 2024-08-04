using Test, GLMakie, Random, Lux

include("../pipeline/symbolic_regression.jl")
include("../architecture/kan_model.jl")
include("../architecture/symbolic_layer.jl")
include("../pipeline/utils.jl")
include("../pipeline/optim_trainer.jl")
include("../pipeline/optimisation.jl")
include("../pipeline/plot.jl")
include("../utils.jl")
using .KolmogorovArnoldNets
using .SymbolicRegression
using .symbolic_layer
using .PipelineUtils
using .OptimTrainer
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
    Legend(fig, ax, position = :rt)
    save("figures/test_sin_fitting.png", fig)
end

function test_lock_symb()
    layer = SymbolicDense(3, 2)
    ps, st = Lux.setup(Random.default_rng(), layer)
    _, layer, ps = lock_symbolic(layer, ps, 3, 2, "sin")

    @test layer.fcns[2][3](2.4) ≈ sin(2.4)
    @test layer.fcn_names[2][3] == "sin"
    @test all(ps[2, 3, :] .≈ [1.0, 0.0, 1.0, 0.0])

    ayer = SymbolicDense(3, 2)
    ps, st = Lux.setup(Random.default_rng(), layer)
    num = 100
    x = range(-1, 1, length=num) |> collect
    noises = randn(num) .* 0.02
    y = 2 .* x .+ 1 .+ noises
    fcn = "x"
    R2, layer, ps = lock_symbolic(layer, ps, 3, 2, fcn; x, y, random=true, seed=123)

    @test layer.fcns[2][3](2.4) ≈ 2.4
    @test layer.fcn_names[2][3] == "x"
    @test ps[2, 3, 1] - 2 < 0.01
    @test ps[2, 3, 2] - 1 < 0.01
    @test ps[2, 3, 3] - 1 < 0.01
    @test ps[2, 3, 4] - 0 < 0.01
    @test R2 >= 0.9
end

function test_suggestion()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    ps, st = Lux.setup(Random.default_rng(), model)

    train_data, test_data = create_data(x -> x[1] * x[2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, normalise_input=false, init_seed=1234)
    opt = create_optim_opt("bfgs", "backtrack")
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=100, verbose=true)
    model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1)
    model, ps, st, best_name, best_fcn, best_R2 = suggest_symbolic(model, ps, st, 1, 1, 1)
    @test best_R2 >= 0.8
end

function test_auto()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    ps, st = Lux.setup(Random.default_rng(), model)

    train_data, test_data = create_data(x -> x[1] + x[2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, normalise_input=false, init_seed=1234)
    opt = create_optim_opt("bfgs", "backtrack")
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=20, verbose=true)
    model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
    y, scales, st = model(train_data[1], ps, st)
    plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=["x1 + x1"], title="KAN", file_name="symbolic_test")
    model, ps, st = prune(Random.default_rng(), model, ps, st)
    y, scales, st = model(train_data[1], ps, st)
    model, ps, st = auto_symbolic(model, ps, st; lib=["sin", "exp", "x^2"])
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=20, verbose=true) # Don't fprget to reinit after pruning!
    model, ps, st = train!(trainer; ps=ps, st=st, λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
    return model, ps, st
end

function test_formula(model, ps, st)
    formula, x0, st = symbolic_formula(model, ps, st)
    println(formula[1])
    formula = string(formula[1])
    return round_formula(formula), st
end

function plot_symb(model, st, form)
    plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[form], title="Pruned Symbolic KAN", file_name="symbolic_test_pruned")
end

@testset "KAN_model Tests" begin
    test_param_fitting()
    test_sin_fitting()
    test_lock_symb()
    test_suggestion()
end

m, p, s = test_auto()
formula, st = test_formula(m, p, s)
plot_symb(m, st, formula)