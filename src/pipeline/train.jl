module Trainer

export init_trainer, train!

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics
# using CUDA, KernelAbstractions

include("utils.jl")
include("../architecture/kan_model.jl")
include("plot.jl")
using .PipelineUtils
using .KolmogorovArnoldNets: fwd!, update_grid!, prune!
using .Plotting

function L2_loss(model, x, y)
    """
    Compute L2 loss between predicted and true values.
    
    Args:
    - model: KAN model.
    - x: input tensor.
    - y: true output tensor.
    
    Returns:
    - loss: L2 loss.
    """
    ŷ = fwd!(model, x)
    return sum((ŷ .- y).^2, dims=2)
end

# Log the loss to CSV
function log_csv(epoch, time, train_loss, test_loss, reg, file_name)
    open(file_name, "a") do file
        write(file, "$epoch,$time,$train_loss,$test_loss,$reg\n")
    end
end

mutable struct trainer
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn
    max_epochs::Int
    verbose::Bool
end

function init_trainer(train_loader, test_loader, optimiser; loss_fn=nothing, max_epochs=100, verbose=true)
    """
    Initialise trainer for training symbolic model.

    Args:
    - model: symbolic model to train.
    - train_loader: training dataloader.
    - test_loader: test dataloader.
    - optimiser: optimiser object.
    - loss_fn: loss function.
    - max_epochs: maximum number of epochs.
    - verbose: whether to print training progress.

    Returns:
    - t: trainer object.
    """
    return trainer(train_loader, test_loader, optimiser, loss_fn, max_epochs, verbose)
end

function train!(t::trainer, model; log_loc="logs/", img_loc="figures/", prune_bool=false, plot=false, plot_mask=false, update_grid_bool=true, grid_update_num=10, stop_grid_update_step=50, reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0)
    """
    Train symbolic model.

    Args:
    - t: trainer object.
    - model: symbolic model to train.

    Returns:
    - model: trained symbolic model.
    """

    if plot_mask
        prune_bool = true
    end

    # Regularisation
    function reg(acts_scale)
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            term1 = ifelse.(x .< th, 1.0, 0.0)
            term2 = ifelse.(x .> th, 1.0, 0.0)
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
        end

        reg_ = 0.0
        for i in eachindex(acts_scale)
            reg_ += sum(abs2, non_linear(acts_scale[i]))
            coeff_l1 = sum(mean(abs.(model.act_fcns[i].coef), dims=2))[1]
            reg_ += λ_l1 * coeff_l1 * λ_coefdiff * λ_coef
        end

        return reg_
    end

    if isnothing(t.loss_fn)
        t.loss_fn = (m, x, y) -> L2_loss(m, x, y) .+ λ*reg.(m.acts_scale)
    end

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_name = log_loc * "log_" * date_str * ".csv"

    # Create csv with header
    open(file_name, "w") do file
        write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation")
    end

    start_time = time()
    for epoch in ProgressBar(1:t.max_epochs)
        train_loss = 0.0
        test_loss = 0.0

        # Training
        Flux.trainmode!(model)
        for (x, y) in t.train_loader
            x, y = x |> permutedims, y |> permutedims
            loss_val, grad = Flux.withgradient(m -> t.loss_fn(m, x, y), model)
            opt.opt_state, m = Optimisers.update(opt.opt_state, m, grad[1])
            train_loss += loss_val
            if (epoch % grid_update_freq == 0) && (epoch < stop_grid_update_step) && update_grid_bool
                update_grid!(model, x)
            end
        end

        opt.LR = opt.LR_scheduler(epoch, opt.LR)

        # Testing
        Flux.testmode!(model)
        for (x, y) in t.test_loader
            x, y = x |> permutedims, y |> permutedims
            test_loss += loss_fn(model, x, y)
        end

        if prune_bool && !plot_mask
            prune!(model)
        end

        train_loss /= length(t.train_loader.data)
        test_loss /= length(t.test_loader.data)

        time_epoch = time() - start_time
        log_csv(epoch, time_epoch, train_loss, test_loss, reg(model.acts_scale), file_name)

        if plot
            plot_kan!(model; folder=img_loc, prune_and_mask=plot_mask)
        end
    end
end

end



