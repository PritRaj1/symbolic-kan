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
    train_data, test_data = create_data(x -> x[1] * x[2], N_var=2, x_range=(-1,1), N_train=100, N_test=100, normalise_input=false, init_seed=1234)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    opt = create_optim_opt(model, "lbfgs", "hagerzhang")
    trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=1e5, verbose=true)
    train!(trainer; λ=0.1, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1)

    @test sum(trainer.state.act_scale) > 0.0
    return trainer, test_data[1]
end

function test_prune(trainer, x)
    model, ps, st = trainer.model, trainer.params, trainer.state
    mask_before = st.mask[1]
    model, ps, st = prune(Random.default_rng(), model, ps, st)
    mask_after = st.mask
    y, st = model(x, ps, st)

    sum_mask_after = 0.0
    for i in eachindex(mask_after)
        sum_mask_after += sum(mask_after[i])
    end

    println("Number of neurons after pruning: ", sum_mask_after)
    @test sum_mask_after != sum(mask_before)
    return model, st
end

function test_plot(model, st)
    plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=["x1 * x2"], title="KAN")
end

t, x = test_trainer()
model, st = test_prune(t, x)
test_plot(model, st)