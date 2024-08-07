ENV["GPU"] = "true"

using Lux, LuxCUDA, CUDA, KernelAbstractions, Tullio, Test, Random, Zygote

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
include("../architecture/kan_model.jl")
include("../pipeline/symbolic_regression.jl")
include("../pipeline/optim_trainer.jl")
include("../pipeline/utils.jl")
include("../pipeline/plot.jl")
include("../pipeline/optimisation.jl")
include("../utils.jl")
using .dense_kan
using .symbolic_layer
using .KolmogorovArnoldNets
using .SymbolicRegression
using .OptimTrainer
using .PipelineUtils
using .Plotting
using .Optimisation
using .Utils: device, round_formula

# Test b_spline_layer
function test_spline_lyr()
    x = randn(100, 3) |> device
    l = KAN_Dense(3, 5) 
    ps = Lux.initialparameters(Random.GLOBAL_RNG, l) |> device
    st = Lux.initialstates(Random.GLOBAL_RNG, l) |> device
    y, st = l(x, ps, st.mask)
    l, grads = Zygote.withgradient(p -> sum(l(x, p, st.mask)[1]), ps)
    @test !isnothing(y)
end

# Test symbolic layer
function test_symb_lyr()
    layer = SymbolicDense(3, 5) 
    x = randn(100, 3) |> device
    ps = Lux.initialparameters(Random.GLOBAL_RNG, layer) |> device
    st = Lux.initialstates(Random.GLOBAL_RNG, layer) |> device
    z, st = layer(x, ps, st.mask)
    l, grads = Zygote.withgradient(p -> sum(layer(x, p, st.mask)[1]), ps)
    @test !isnothing(l)
end

function test_model()
    Random.seed!(123)
    model = KAN_model([2,5,3]; k=3, grid_interval=5) 
    ps = Lux.initialparameters(Random.default_rng(), model) |> device
    st = Lux.initialstates(Random.default_rng(), model) |> device

    x = randn(Float32, 100, 2) |> device
    y, st = model(x, ps, st)
    l, grads = Zygote.withgradient(p -> sum(model(x, p, st)[1]), ps)
    @test all(size(y) .== (100, 3))
    @test !isnothing(l)
end

function test_grid()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5) 
    ps = Lux.initialparameters(Random.default_rng(), model) |> device
    st = Lux.initialstates(Random.default_rng(), model) |> device

    before = model.act_fcns[1].grid[1, :]
    
    x = randn(Float32, 100, 2) .* 5 |> device
    model, ps = update_grid(model, x, ps, st)
    after = model.act_fcns[1].grid[1, :]
    @test abs(sum(before) - sum(after)) > 0.1
end

function test_training()
    train_data, test_data = create_data(x -> x[:,1] .* x[:,2], N_var=2, x_range=(-1,1), N_train=10, N_test=10, normalise_input=false, init_seed=1234)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    opt = create_optim_opt("bfgs", "backtrack")
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=3, verbose=true)
    model, params, state = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=2.0, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=15)

    # check loss
    x, y = train_data
    ŷ, state = model(x, params, state)
    state = cpu_device()(state) 
    y = y |> device
    loss = sum((ŷ .- y).^2)
    println("Loss: ", loss)

    @test sum(state[Symbol("act_scale_1")]) > 0.0
    plot_kan(model, state; mask=true, in_vars=["x1", "x2"], out_vars=["x1 * x2"], title="KAN", file_name="gpu_test")
    return model, params, state, x
end

function test_prune(model, ps, st, x)
    mask_before = st[Symbol("mask_2")]
    model, ps, st = prune(Random.default_rng(), model, ps, st)
    mask_after =  st[Symbol("mask_2")]
    y, st = model(x, ps, st)
    st = cpu_device()(st)

    sum_mask_after = 0.0
    for i in eachindex(mask_after)
        sum_mask_after += sum(mask_after[i])
    end

    println("Number of neurons after pruning: ", sum_mask_after)
    @test sum_mask_after != sum(mask_before)
    return model, ps, st
end

function test_auto_symbolic(model, ps, st)
    model, ps, st = auto_symbolic(model, ps, st; lib=["sin", "exp", "x^2"])
    return model, ps, st
end

function test_formula(model, ps, st)
    formula, x0, st = symbolic_formula(model, ps, st)
    formula = string(formula[1])
    formula = round_formula(formula)
    println("Formula: ", formula)
    return formula, st
end

function plot_symb(model, st, form)
    plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[form], title="Pruned Symbolic KAN", file_name="gpu_symbolic_test")
end

@testset "KAN Tests" begin
    test_spline_lyr()
    test_symb_lyr()
    test_model()
    test_grid()
end

model, ps, st, x = test_training()
model, ps, st = test_prune(model, ps, st, x)
model, ps, st = test_auto_symbolic(model, ps, st)
formula, st = test_formula(model, ps, st)
plot_symb(model, st, formula)