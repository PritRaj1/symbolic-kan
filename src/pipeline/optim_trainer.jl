module OptimTrainer

export init_optim_trainer, train!

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics, Optim, Zygote

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
using .PipelineUtils: log_csv, L2_loss!, diff3
using .KolmogorovArnoldNets: fwd!, update_grid!
using .Optimisation: opt_get

veclength(params::Flux.Params) = sum(length, params.params)
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))

mutable struct optim_trainer
    model
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn!
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

function train!(t::optim_trainer; log_loc="logs/", update_grid_bool=true, grid_update_num=10, stop_grid_update_step=50, reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0)
    """
    Train symbolic model.

    Args:
    - t: trainer object.

    Returns:
    - model: trained model.
    """
    λ = Float32(λ)
    λ_l1 = Float32(λ_l1)
    λ_entropy = Float32(λ_entropy)
    λ_coef = Float32(λ_coef)
    λ_coefdiff = Float32(λ_coefdiff)

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
            entropy = -1 * sum(p .* log.(p .+ 1e-3f0))
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
        println(typeof(l2))
        reg_ = reg(m)
        reg_ = λ * reg_
        return l2 .+ reg_
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
    epoch = 0

    # All x for gird update
    x_collection = zeros(Float32, size(first(t.train_loader)[1])[1], 0)
    for (x, y) in t.train_loader
        x_collection = hcat(x_collection, x)
    end
    x_collection = x_collection |> permutedims

    # Create manual dataloaders from Flux.Data.DataLoader
    train_loader = [(x, y) for (x, y) in t.train_loader]
    test_loader = [(x, y) for (x, y) in t.test_loader]

    # Training
    function batch_train!(m)
        train_loss = 0.0

        batch_step = 1
        for (x, y) in train_loader
            x, y = x |> permutedims, y |> permutedims
            train_loss += t.loss_fn!(m, x, y)
            train_loss = train_loss / batch_step
            batch_step += 1
        end

        return train_loss
    end

    # Evaluating callback
    function log_callback(state)
        train_loss = 0.0
        test_loss = 0.0

        for (x, y) in train_loader
            x, y = x |> permutedims, y |> permutedims
            train_loss += t.loss_fn!(t.model, x, y)
        end

        for (x, y) in test_loader
            x, y = x |> permutedims, y |> permutedims
            test_loss += t.loss_fn!(t.model, x, y)
        end

        if (epoch % grid_update_freq == 0) && (epoch < stop_grid_update_step) && update_grid_bool
            update_grid!(t.model, x_collection)
        end

        train_loss = train_loss / length(train_loader)
        test_loss = test_loss / length(test_loader)

        reg_ = reg(t.model)
        log_csv(epoch, time() - start_time, train_loss, test_loss, reg_, file_name; log_time=t.log_time)
        
        epoch += 1

        return false
    end

    # From https://github.com/baggepinnen/FluxOptTools.jl
    function get_fg(loss)
        pars = Flux.params(t.model)
        grads = Zygote.gradient(loss, t.model)
        p0 = zeros(pars)
        copy!(p0, pars)
        fg! = function (F,G,w)
            copy!(pars, w)
            Flux.loadparams!(t.model, pars)
            if isnothing(G)
                l, back = Zygote.pullback(loss, t.model)
                grads = back(1.0)
                copy!(G, grads)
                return l
            end
            if !isnothing(F)
                return loss(t.model)
            end
        end
        return fg!, p0
    end

    fg!, p0 = get_fg((m) -> batch_train!(m))
    res = Optim.optimize(Optim.only_fg!(fg!), p0, opt_get(t.opt), Optim.Options(show_trace=true, iterations=t.max_epochs, callback=log_callback, x_abstol=1e-8, f_abstol=1e-8, g_abstol=1e-8))
    _, re = Flux.destructure(t.model)
    Flux.loadmodel!(t.model, re(res.minimizer))
end

end