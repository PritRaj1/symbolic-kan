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
    train_loader, test_loader = create_loaders(x -> x^2, N_var=2, x_range=(-1,1), N_train=100, N_test=100, batch_size=10, normalise_x=true, normalise_y=true, init_seed=1234)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    lr_scheduler = step_decay_scheduler(5, 0.8, 1e-6)
    opt = create_opt(model, "adam"; LR=0.0001, decay_scheduler=lr_scheduler)
    trainer = init_trainer(model, train_loader, test_loader, opt; max_epochs=30, verbose=true)
    train!(trainer)
    return trainer.model
end

function test_prune(model)
    return prune(model)
end

function test_plot(model)
    plot_kan!(model; mask=true, in_vars=["x1", "x2"], out_vars=["y1", "y2"], title="KAN")
end


model = test_trainer()
# model = KAN([2,5,1]; k=3, grid_interval=5)
model = test_prune(model)
test_plot(model)
