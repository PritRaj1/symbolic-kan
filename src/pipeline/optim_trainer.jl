module OptimTrainer

export init_optim_trainer, train!

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics, Optim, Zygote

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
using .PipelineUtils: log_csv, L2_loss!
using .KolmogorovArnoldNets: fwd!, update_grid!
using .Optimisation: opt_get

veclength(params::Flux.Params) = sum(length, params.params)
veclength(grads::Union{Dict,NamedTuple}) = sum(length, values(grads))
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))
Base.zeros(grads::Union{Dict,NamedTuple}) = zeros(veclength(grads))

mutable struct optim_trainer
    model
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn
    max_epochs::Int
    verbose::Bool
    log_time::Bool
end

function init_optim_trainer(model, train_loader, test_loader, optim_optimiser; loss_fn=nothing, max_epochs=100, verbose=true, log_time=true)
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
    return optim_trainer(model, train_loader, test_loader, optim_optimiser, loss_fn, max_epochs, verbose, log_time)
end

function train!(t::optim_trainer; log_loc="logs/", update_grid_bool=true, grid_update_num=50, stop_grid_update_step=50, reg_factor=1.0, mag_threshold=1e-16, 
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
        t.log_time ? write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n") : write(file, "Epoch,Train Loss,Test Loss,Regularisation\n")
    end

    start_time = time()
    num_steps = t.max_epochs * length(t.train_loader.data)
    epoch = 0

    # Create manual dataloaders from Flux.Data.DataLoader
    train_loader = [(x, y) for (x, y) in t.train_loader]
    test_loader = [(x, y) for (x, y) in t.test_loader]

    params = Flux.params(t.model)

    # Training
    function batch_train!(m)
        train_loss = 0.0

        Flux.trainmode!(m)
        for (x, y) in train_loader
            x, y = x |> permutedims, y |> permutedims
            train_loss += t.loss_fn(m, x, y)
        end

        train_loss = train_loss / length(train_loader)

        return train_loss
    end

    # Evaluating callback
    function log_callback()
        train_loss = 0.0
        test_loss = 0.0
        Flux.testmode!(t.model)

        for (x, y) in train_loader
            x, y = x |> permutedims, y |> permutedims
            train_loss += t.loss_fn(t.model, x, y)

            if (num_steps % grid_update_freq == 0) && (num_steps < stop_grid_update_step) && update_grid_bool
                update_grid!(t.model, x)
            end
        end

        for (x, y) in test_loader
            x, y = x |> permutedims, y |> permutedims
            test_loss += t.loss_fn(t.model, x, y)
        end

        train_loss = train_loss / length(train_loader)
        test_loss = test_loss / length(test_loader)

        reg_ = reg(t.model.act_scale)
        log_csv(epoch, time() - start_time, train_loss, test_loss, reg_, file_name; log_time=t.log_time)
        
        epoch += 1
    end

    function optfuns(pars::Union{Flux.Params, Zygote.Params})
        p0 = zeros(pars)
        copy!(p0, pars)
        fg! = function (F,G,w)
            copy!(pars, w)
            Flux.loadparams!(t.model, pars)
            if !isnothing(G)
                l, back = Zygote.withgradient(m -> batch_train!(m), t.model)
                grads = back[1]
                G = zeros(grads)
                copy!(G, grads)
                return l
            end
            if !isnothing(F)
                return batch_train!(t.model)
            end
        end
        return fg!, p0
    end

    Zygote.refresh()
    fg!, p0 = optfuns(params)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, opt_get(t.opt), Optim.Options(show_trace=true, iterations=t.max_epochs, callback=log_callback))
    # Flux.loadparams!(t.model, res.minimizer)
    copy!(params, res.minimizer)
    Flux.loadparams!(t.model, params)
end

end