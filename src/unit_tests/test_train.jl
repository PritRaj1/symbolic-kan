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
    train_loader, test_loader = create_loaders(x -> x, N_var=2, x_range=(-1,1), N_train=10, N_test=10, batch_size=2, normalise_x=false, normalise_y=false, init_seed=1234)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    lr_scheduler = step_decay_scheduler(1, 0.1, 1e-4)
    opt = create_opt(model, "adam"; LR=0.01, decay_scheduler=lr_scheduler)
    trainer = init_trainer(train_loader, test_loader, opt; max_epochs=2, verbose=true)
    train!(trainer, model)
    return model
end

function test_prune(model)
    prune(model)
end

function test_plot(model)
    x = randn(100, 2)
    y = fwd!(model, x)
    plot_kan!(model; prune_and_mask=false)
end


# model = test_trainer()
model = KAN([2,5,1]; k=3, grid_interval=5)
# test_prune(model)
test_plot(model)
