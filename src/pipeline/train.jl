module Trainer

export init_trainer, train!, L2_loss

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics, Optimisers
# using CUDA, KernelAbstractions

include("utils.jl")
include("../architecture/kan_model.jl")
include("plot.jl")
using .PipelineUtils
using .KolmogorovArnoldNets: fwd!, update_grid!
using .Plotting

function L2_loss!(model, x, y)
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
    return sum((ŷ .- y).^2)
end

# Log the loss to CSV
function log_csv(epoch, time, train_loss, test_loss, reg, file_name)
    open(file_name, "a") do file
        write(file, "$epoch,$time,$train_loss,$test_loss,$reg\n")
    end
end

mutable struct trainer
    model
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn
    max_epochs::Int
    verbose::Bool
end

function init_trainer(model, train_loader, test_loader, optimiser; loss_fn=nothing, max_epochs=100, verbose=true)
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
    return trainer(model, train_loader, test_loader, optimiser, loss_fn, max_epochs, verbose)
end

function train!(t::trainer; log_loc="logs/", update_grid_bool=true, grid_update_num=50, stop_grid_update_step=50, reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0)
    """
    Train symbolic model.

    Args:
    - t: trainer object.

    Returns:
    - model: trained model.
    """

    # Regularisation
    function reg(acts_scale)
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            term1 = ifelse.(x .< th, 1.0, 0.0)
            term2 = ifelse.(x .> th, 1.0, 0.0)
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
        end

        reg_ = 0.0
        for i in eachindex(acts_scale[:, 1, 1])
            reg_ += sum(abs2, non_linear(acts_scale[i, :, :]))
            coeff_l1 = sum(mean(abs.(t.model.act_fcns[i].coef), dims=2))
            reg_ += λ_l1 * coeff_l1 * λ_coefdiff * λ_coef
        end

        return reg_
    end

    # l1 rregularisation loss
    function reg_loss!(m, x, y)
        l2 = L2_loss!(m, x, y)
        return mean(l2 .+ λ * reg(m.act_scale))
    end

    if isnothing(t.loss_fn)
        t.loss_fn = reg_loss!
    end

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    # Create folders
    !isdir(log_loc) && mkdir(log_loc)
    
    # Create csv with header
    file_name = log_loc * "log_" * date_str * ".csv"
    open(file_name, "w") do file
        write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n")
    end

    start_time = time()
    num_steps = t.max_epochs * length(t.train_loader.data)
    for epoch in ProgressBar(1:t.max_epochs)
        train_loss = 0.0
        test_loss = 0.0

        # Training
        Flux.trainmode!(t.model)
        for (x, y) in t.train_loader
            x, y = x |> permutedims, y |> permutedims
            
            loss_val, grad = Flux.withgradient(m -> t.loss_fn(m, x, y), t.model)
            t.opt.opt_state, t.model = Optimisers.update(t.opt.opt_state, t.model, grad[1])
            train_loss += loss_val

            if (num_steps % grid_update_freq == 0) && (num_steps < stop_grid_update_step) && update_grid_bool
                update_grid!(t.model, x)
            end
        end

        t.opt.LR = t.opt.LR_scheduler(epoch, t.opt.LR)
        Optimisers.adjust!(t.opt.opt_state, t.opt.LR)
        
        # Testing
        Flux.testmode!(t.model)
        for (x, y) in t.test_loader
            x, y = x |> permutedims, y |> permutedims
            test_loss += t.loss_fn(t.model, x, y)
        end

        train_loss /= length(t.train_loader.data)
        test_loss /= length(t.test_loader.data)

        time_epoch = time() - start_time
        log_csv(epoch, time_epoch, train_loss, test_loss, mean(reg(t.model.act_scale)), file_name)

        if t.verbose
            println("Epoch: $epoch, Train Loss: $train_loss, Test Loss: $test_loss, Regularisation: $(reg(t.model.act_scale))")
        end
    end
end

end



