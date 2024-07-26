using Test

include("../pipeline/train.jl")
include("../pipeline/utils.jl")
include("../pipeline/plot.jl")
include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets
using .Trainer
using .PipelineUtils
using .Plotting

function test_trainer()
    train_loader, test_loader = create_loaders(x -> x^2, N_var=2, x_range=(-1,1), N_train=500, N_test=500, batch_size=50, normalise_x=false, normalise_y=false, init_seed=1234)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    lr_scheduler = step_decay_scheduler(5, 0.8, 1e-3)
    opt = create_opt(model, "adam"; LR=0.1, decay_scheduler=lr_scheduler)
    trainer = init_trainer(model, train_loader, test_loader, opt; max_epochs=30, verbose=true)
    train!(trainer)

    @test sum(mode.act_scale) > 0.0
    return trainer.model
end

function test_prune(model)
    mask_before = model.mask
    model = prune(model)
    mask_after = model.mask

    println("Number of parameters before pruning: ", sum(mask_before))
    println("Number of parameters after pruning: ", sum(mask_after))
    @test sum(mask_after) < sum(mask_before)
    return model
end

function test_plot(model)
    plot_kan!(model; mask=true, in_vars=["x1", "x2"], out_vars=["y1", "y2"], title="KAN")
end


model = test_trainer()
# model = KAN([2,5,1]; k=3, grid_interval=5)
model = test_prune(model)
test_plot(model)
