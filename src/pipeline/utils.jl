module PipelineUtils

export create_loaders, create_opt, step!

using Flux, Optimisers, Statistics, Random
# using CUDA, KernelAbstractions

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

### Step LR scheduler ### 
struct decay_scheduler
    step::Int
    decay::Float64
    min_LR::Float64
end

function step_decay_scheduler(step, decay, min_LR)
    return decay_scheduler(step, decay, min_LR)
end

function (s::decay_scheduler)(epoch, LR)
    return max(LR * s.decay^(epoch // s.step), s.min_LR)
end

### Optimiser ###
optimiser_map = Dict(
    "adam" => Optimisers.Adam,
    "sgd" => Optimisers.Descent
)

mutable struct optimiser_state
    opt_state
    LR_scheduler::Function
    LR::Float32
end

function create_opt(model, type="adam"; LR=0.01, schedule_LR=false, step=10, γ=0.1, min_LR=0.001)
    """
    Create optimiser.

    Args:
    - type: optimiser to use.
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

    opt = optimiser_map[type](LR)
    opt_state = Optimisers.setup(opt, model)

    return optimiser_state(opt_state, schedule_fcn, LR)
end

end
