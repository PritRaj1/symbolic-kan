module OptimTrainer

export init_optim_trainer, train!

using Flux, ProgressBars, Dates, Tullio, CSV, Statistics, Optim, Zygote#, FluxOptTools

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
using .PipelineUtils: log_csv, L2_loss!, diff3
using .KolmogorovArnoldNets: fwd!, update_grid!
using .Optimisation: opt_get

veclength(params::Flux.Params) = sum(length, params.params)
veclength(grads::Union{Dict, NamedTuple}) = return sum(length(grads[p]) for p in keys(grads) if grads[p] !== nothing)
Base.zeros(pars::Flux.Params) = zeros(veclength(pars))
Base.zeros(grads::Union{Dict, NamedTuple}) = zeros(veclength(grads))

mutable struct optim_trainer
    model
    train_loader::Flux.Data.DataLoader
    test_loader::Flux.Data.DataLoader
    opt
    loss_fn!
    epoch::Int
    max_epochs::Int
    update_grid_bool::Bool
    verbose::Bool
    log_time::Bool
end

function init_optim_trainer(model, train_loader, test_loader, optim_optimiser; loss_fn=nothing, max_epochs=100, update_grid_bool=true, verbose=true, log_time=true)
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
    return optim_trainer(model, train_loader, test_loader, optim_optimiser, loss_fn, 0, max_epochs, update_grid_bool, verbose, log_time)
end

function train!(t::optim_trainer; log_loc="logs/", grid_update_num=10, stop_grid_update_step=50, reg_factor=1.0, mag_threshold=1e-16, 
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

    # No batching for optim please
    x_train = zeros(Float32, size(first(t.train_loader)[1])[1], 0)
    y_train = zeros(Float32, size(first(t.train_loader)[2])[1], 0)
    for (x, y) in t.train_loader
        x_train = hcat(x_train, x)
        y_train = hcat(y_train, y)
    end
    x_train = x_train |> permutedims
    y_train = y_train |> permutedims

    x_test = zeros(Float32, size(first(t.test_loader)[1])[1], 0)
    y_test = zeros(Float32, size(first(t.test_loader)[2])[1], 0)
    for (x, y) in t.test_loader
        x_test = hcat(x_test, x)
        y_test = hcat(y_test, y)
    end
    x_test = x_test |> permutedims
    y_test = y_test |> permutedims

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)

    # Regularisation
    function reg(m)
        acts_scale = m.act_scale
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            term1 = ifelse.(x .< th, Float32(1), Float32(0))
            term2 = ifelse.(x .>= th, Float32(1), Float32(0))
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
        end

        reg_ = Float32(0.0)
        for i in eachindex(acts_scale[:, 1, 1])
            vec = reshape(acts_scale[i, :, :], :)
            p = vec ./ sum(vec)
            l1 = sum(non_linear(vec))
            entropy = -1 * sum(p .* log.(p .+ Float32(1e-3)))
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
    function reg_loss!(m, x, y; epoch=0)
        
        # Update grid once per epoch if it's time
        if (epoch % grid_update_freq == 0) && (epoch < stop_grid_update_step) && t.update_grid_bool
            update_grid!(t.model, x)
        end

        l2 = L2_loss!(m, x, y)
        reg_ = reg(m)
        reg_ = λ * reg_
        return l2 .+ reg_
    end

    if isnothing(t.loss_fn!)
        t.loss_fn! = reg_loss!
    end

    # Create folders
    !isdir(log_loc) && mkdir(log_loc)
    
    # Create csv with header
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_name = log_loc * "log_" * date_str * ".csv"
    open(file_name, "w") do file
        t.log_time ? write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n") : write(file, "Epoch,Train Loss,Test Loss,Regularisation\n")
    end

    start_time = time()

    # Training
    function train_loss!(m)
        return t.loss_fn!(m, x_train, y_train; t.epoch)
    end

    # Evaluating callback
    function log_callback(state)
        t.update_grid_bool = false

        train_loss = t.loss_fn!(t.model, x_train, y_train; t.epoch)
        test_loss = t.loss_fn!(t.model, x_test, y_test; t.epoch)
        reg_ = reg(t.model)
        
        log_csv(t.epoch, time() - start_time, train_loss, test_loss, reg_, file_name; log_time=t.log_time)
        
        t.epoch += 1
        t.update_grid_bool = true

        return false
    end

    function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}})
        keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
        return NamedTuple{keepnames}(namedtuple)
    end

    # # From https://github.com/baggepinnen/FluxOptTools.jl
    # function get_fg!(loss, model)
    #     pars = Flux.trainable(model)
    #     p0, re = Flux.destructure(pars)

    #     function fg!(F,G,w)
    #         copy!(p0, w)
    #         Flux.loadmodel!(t.model, re(w))
    #         if !isnothing(G)
    #             l, grads = Flux.withgradient(loss, t.model)
    #             grads = grads[1]

    #             trainables = Flux.trainable(t.model)
    #             drop_keys = ()
    #             for key in keys(grads)
    #                 if key ∉ trainables
    #                     drop_keys = (drop_keys..., :key)
    #                 else
    #                     trainables_ = Flux.trainable(t.model.:key)
    #                     drop_keys_ = ()
    #                     for key_ in keys(grads[key])
    #                         if key_ ∉ trainables_
    #                             drop_keys_ = (drop_keys_..., :key_)
    #                         end
    #                     end
    #                     grads[key][key_] = dropnames(grads[key][key_], drop_keys_)
    #                 end
    #             end

    #             grads = dropnames(grads, drop_keys)
    #             println(grads)
    #             println("=====================================================")
    #             println(pars)
    #             grads = Flux.destructure(grads)[1]
    #             copy!(G, grads)
    #             return l
    #         end
    #         if !isnothing(F)
    #             return loss(t.model)
    #         end
    #     end
    #     return fg!, p0
    # end

    # fg!, p0 = get_fg!((m) -> train_loss!(m), t.model)

    # println(fg!(nothing, nothing, p0))

    function optfuns(loss, pars::Union{Flux.Params, Zygote.Params})
        grads = Zygote.gradient(loss, pars)
        p0 = zeros(pars)
        copy!(p0, pars)
        gradfun = function (g,w)
            copy!(pars, w)
            grads = Zygote.gradient(loss, pars)
            copy!(g, grads)
        end
        lossfun = function (w)
            copy!(pars, w)
            loss()
        end
        fg! = function (F,G,w)
            copy!(pars, w)
            if !isnothing(G)
                l, back = Zygote.withgradient(loss, pars)
                grads = back[1]
                copy!(G, grads)
                return l
            end
            if !isnothing(F)
                return loss()
            end
        end
        lossfun, gradfun, fg!, p0
    end

    params = Flux.params(Flux.trainables(t.model))
    l, grads = Zygote.withgradient(() -> train_loss!(t.model), params)
    println(grads[1])
    _, _, fg!, p0 = optfuns(() -> train_loss!(t.model), params)

    res = Optim.optimize(Optim.only_fg!(fg!), p0, opt_get(t.opt), Optim.Options(show_trace=true, iterations=t.max_epochs, callback=log_callback))
    params = Flux.params(t.model)
    params = copy!(params, res.minimizer)
    Flux.loadparams!(t.model, params)
end

end