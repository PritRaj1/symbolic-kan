module FluxTrainer

export init_flux_trainer, train!

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics, Optimisers
# using CUDA, KernelAbstractions

include("utils.jl")
include("../architecture/kan_model.jl")
using .PipelineUtils: log_csv, L2_loss!, diff3
using .KolmogorovArnoldNets: fwd!, update_grid!

mutable struct flux_trainer
    model
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn!
    max_epochs::Int
    verbose::Bool
    log_time::Bool
end

function init_flux_trainer(model, train_loader, test_loader, flux_optimiser; loss_fn=nothing, max_epochs=100, verbose=true, log_time=true)
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
    return flux_trainer(model, train_loader, test_loader, flux_optimiser, loss_fn, max_epochs, verbose, log_time)
end

function train!(t::flux_trainer; log_loc="logs/", update_grid_bool=true, grid_update_num=1000, stop_grid_update_step=5000, reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0)
    """
    Train symbolic model.

    Args:
    - t: trainer object.

    Returns:
    - model: trained model.
    """

    # Regularisation
    function reg(m)
        acts_scale = m.act_scale
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            term1 = ifelse.(x .< th, Float32(1.0), Float32(0.0))
            term2 = ifelse.(x .> th, Float32(1.0), Float32(0.0))
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
        end

        reg_ = 0.0
        for i in eachindex(acts_scale[:, 1, 1])
            vec = reshape(acts_scale[i, :, :], :)
            p = vec ./ sum(vec)
            l1 = sum(non_linear(vec))
            entropy = -1 * sum(p .* log.(p .+ 1e-3))
            reg_ += (l1 * λ_l1) + (entropy * λ_entropy)
        end

        for i in eachindex(m.act_fcns)
            coeff_l1 = sum(mean(abs.(m.act_fcns[i].coef), dims=2))
            coeff_diff_l1 = sum(mean(abs.(diff3(m.act_fcns[i].coef)), dims=2))
            reg_ += (λ_coef * coeff_l1) + (λ_coefdiff * coeff_diff_l1)
        end

        return reg_
    end

    # l1 regularisation loss
    function reg_loss!(m, x, y)
        l2 = L2_loss!(m, x, y)
        reg_ = reg(m)
        reg_ = λ * reg_
        loss = mean(l2 .+ reg_)
        return loss
    end

    if isnothing(t.loss_fn!)
        t.loss_fn! = reg_loss!
    end

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

    # Create folders
    !isdir(log_loc) && mkdir(log_loc)
    
    # Create csv with header
    file_name = log_loc * "log_" * date_str * ".csv"
    open(file_name, "w") do file
        t.log_time ? write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n") : write(file, "Epoch,Train Loss,Test Loss,Regularisation\n")
    end

    start_time = time()
    num_steps = t.max_epochs * length(t.train_loader.data)
    step = 0
    for epoch in ProgressBar(1:t.max_epochs)
        train_loss = 0.0
        test_loss = 0.0

        # Training
        Flux.trainmode!(t.model)
        for (x, y) in t.train_loader
            x, y = x |> permutedims, y |> permutedims
            
            loss_val, grad = Flux.withgradient(m -> t.loss_fn!(m, x, y), t.model)
            t.opt.opt_state, t.model = Optimisers.update(t.opt.opt_state, t.model, grad[1])
            train_loss += loss_val

            if (step % grid_update_freq == 0) && (step < stop_grid_update_step) && update_grid_bool
                update_grid!(t.model, x)
            end
            step += 1
        end

        t.opt.LR = t.opt.LR_scheduler(epoch, t.opt.LR)
        Optimisers.adjust!(t.opt.opt_state, t.opt.LR)
        
        # Testing
        Flux.testmode!(t.model)
        for (x, y) in t.test_loader
            x, y = x |> permutedims, y |> permutedims
            test_loss += t.loss_fn!(t.model, x, y)
        end

        train_loss /= length(t.train_loader.data)
        test_loss /= length(t.test_loader.data)

        time_epoch = time() - start_time
        reg_val = reg(t.model)
        log_csv(epoch, time_epoch, train_loss, test_loss, reg_val, file_name; t.log_time)

        if t.verbose
            println("Epoch: $epoch, Train Loss: $train_loss, Test Loss: $test_loss, Regularisation: $reg_val")
        end
    end
end

end



