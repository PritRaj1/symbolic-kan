using Test, Random

include("../pipeline/optim_trainer.jl")
include("../pipeline/utils.jl")
include("../pipeline/plot.jl")
include("../architecture/kan_model.jl")
include("../pipeline/optimisation.jl")
using .KolmogorovArnoldNets: KAN_model, KAN, prune
using .OptimTrainer
using .PipelineUtils
using .Plotting
using .Optimisation

function test_trainer()
    train_data, test_data = create_data(x -> x[:,1] .* x[:,2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, normalise_input=false, init_seed=1234)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    opt = create_optim_opt("bfgs", "backtrack")
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=10, verbose=true)
    model, params, state = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=15)

    # check loss
    x, y = train_data
    ŷ, state = model(x, params, state)
    loss = sum((ŷ .- y).^2)
    println("Loss: ", loss)

    @test sum(state[Symbol("act_scale_1")]) > 0.0
    plot_kan(model, state; mask=true, in_vars=["x1", "x2"], out_vars=["x1 * x2"], title="KAN", file_name="kan")
    return model, params, state, train_data[1]
end

function test_prune(model, ps, st, x)
    mask_before = st[Symbol("mask_2")]
    model, ps, st = prune(Random.default_rng(), model, ps, st)
    mask_after = st[Symbol("mask_2")]
    y, st = model(x, ps, st)

    println("Number of neurons after pruning: ", sum(mask_after))
    # @test sum(mask_after) != sum(mask_before)
    return model, st
end

function test_plot(model, st)
    plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=["x1 * x2"], title="Pruned KAN", file_name="kan_pruned")
end

m, p, s, x = test_trainer()
model, st = test_prune(m, p, s, x)
test_plot(model, st)