module PipelineUtils

export create_loaders, create_optimiser

using Flux, CUDA, KernelAbstractions, Optim, Statistics, Random, LineSearches

function create_loaders(fcn; N_var=2, x_range=(-1,1), N_train=1000, N_test=1000, batch_size=32, normalise_x=false, normalise_y=false, init_seed=nothing)
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
    X_train = rand(x_range[1]:x_range[2], N_var, N_train)
    y_train = fcn.(X_train)
    X_test = rand(x_range[1]:x_range[2], N_var, N_test)
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

# Step LR scheduler 
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

function create_optimiser(opt="lbfgs"; history=100, line_search="strong_wolfe", c1=1e-4, c_2=0.9, ρ=2.0, schedule_LR=false, LR=nothing, step=nothing, decay=nothing, min_LR=nothing)
    """
    Create optimiser.

    Args:
    - opt: optimiser to use.
    - schedule_LR: whether to schedule learning rate.
    - LR: learning rate.
    - step: step size for LR scheduler.
    - decay: decay rate for LR scheduler.
    - min_LR: minimum LR for LR scheduler.

    Returns:
    - optimiser: optimiser.
    """

    if line_search == "strong_wolfe"
        line_search = LineSearches.StrongWolfe(c_1=c1, c_2=c_2, ρ=ρ)
    elseif line_search == "hager_zhang"
        line_search = LineSearches.HagerZhang()
    elseif line_search == "more_thuente"
        line_search = LineSearches.MoreThuente()
    else
        line_search = LineSearches.Static()
    end

    if opt == "lbfgs"
        optimiser = Optim.LBFGS(; linesearch=line_search, m=history)
    else opt == "adam"
        optimiser = Flux.ADAM(LR)
    end

    if schedule_LR
        schedule_fcn = step_decay_scheduler(step, decay, min_LR)
    else
        schedule_fcn = (epoch, LR) -> LR
    end

    return optimiser, schedule_fcn
end

end
