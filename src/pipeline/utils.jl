module PipelineUtils

export create_loaders, create_opt, step!

using Flux, CUDA, KernelAbstractions, Optim, Statistics, Random, Zygote, FluxOptTools

include("opt_tools.jl")
using .OptTools: line_search_map, optimiser_map, step_decay_scheduler

### Data loaders ###
function create_loaders(fcn; N_var=2, x_range=(-1.0,1.0), N_train=1000, N_test=1000, batch_size=32, normalise_x=false, normalise_y=false, init_seed=nothing)
    """
    Create train and test dataloaders

    Args:
    - fcn: symbolic function to generate data for.
    - N_var: number of input variables.
    - range: range of input variables.
    - N_train: number of training samples.
    - N_test: number of test samples.
    - normalise_input: whether to normalise input.
    - normalise_output: whether to normalise output.
    - init_seed: random seed.

    Returns:
    - train_loader: training dataloader.
    - test_loader: test dataloader.
    """

    Random.seed!(init_seed)

    # Generate data
    X_train = randn(Float32, (N_var, N_train)) .* (x_range[2] - x_range[1]) .+ x_range[1]
    y_train = fcn.(X_train)
    X_test = randn(Float32, (N_var, N_test)) .* (x_range[2] - x_range[1]) .+ x_range[1]
    y_test = fcn.(X_test)

    # Normalise data
    if normalise_x
        X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
        X_test = (X_test .- mean(X_test, dims=1)) ./ std(X_test, dims=1)
    end

    if normalise_y
        y_train = (y_train .- mean(y_train, dims=1)) ./ std(y_train, dims=1)
        y_test = (y_test .- mean(y_test, dims=1)) ./ std(y_test, dims=1)
    end

    # Create dataloaders
    train_loader = Flux.Data.DataLoader((X_train, y_train); batchsize=batch_size, shuffle=true)
    test_loader = Flux.Data.DataLoader((X_test, y_test); batchsize=batch_size, shuffle=true)

    return train_loader, test_loader
end

mutable struct optimiser
    OPT
    LR_scheduler::Function
    LR::Float32
end

function create_opt(type="lbfgs"; history=100, line_search="strong_wolfe", c1=1e-4, c2=0.9, ρ=2.0, LR=0.01, schedule_LR=false, step=10, decay=0.1, min_LR=0.001)
    """
    Create optimiser.

    Args:
    - type: optimiser to use.
    - history: history size for LBFGS.
    - line_search: line search method.
    - c1: c1 parameter for StrongWolfe.
    - c2: c2 parameter for StrongWolfe.
    - ρ: ρ parameter for StrongWolfe.
    - schedule_LR: whether to schedule learning rate.
    - LR: learning rate.
    - step: step size for LR scheduler.
    - decay: decay rate for LR scheduler.
    - min_LR: minimum LR for LR scheduler.

    Returns:
    - optimiser: optimiser.
    """
    
    if schedule_LR
        schedule_fcn = step_decay_scheduler(step, decay, min_LR)
    else
        schedule_fcn = (epoch, LR) -> LR
    end

    line_search = line_search_map[line_search](c1, c2, ρ)
    opt = optimiser_map[type](line_search, history)

    return optimiser(opt, schedule_fcn, LR)
end

function step!(opt, model, loss_fcn, epoch, x, y; tol=1e-32)
    """
    Perform one step of optimisation.

    Args:
    - opt: optimiser.
    - model: model to optimise.
    - loss_fcn: loss function.
    - epoch: current epoch.
    - x: input data.
    - y: target data.
    - LR: learning rate.

    Returns:
    - loss: loss value.
    """
    
    # Function to optimiser w.r.t params
    function loss()
        return loss_fcn(model, x, y)
    end

    Zygote.refresh()
    params = Flux.params(model)
    lossfun, gradfun, fg!, p0 = optfuns(loss, params)   
    println("Epoch: $epoch, Loss: $(loss())") 
    # grad=Zygote.gradient(loss, p0)
    println("grad", grad)
    optimiser = opt.OPT(opt.LR)
    println("optimiser created", optimiser)
    println("params", p0)
    println("loss", loss())
    results = Optim.optimize(Optim.only_fg!(fg!), p0, optimiser, Optim.Options(iterations=1, show_trace=true, x_abstol=tol, f_abstol=tol, g_abstol=tol))
    opt.LR = opt.LR_scheduler(epoch, opt.LR)

    Flux.loadparams!(model, Optim.minimizer(result))
    println("params loaded")

    return loss()
end

end
