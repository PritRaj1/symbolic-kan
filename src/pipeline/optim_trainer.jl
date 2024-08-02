module OptimTrainer

export init_optim_trainer, train!

using Lux, ProgressBars, Dates, Tullio, CSV, Statistics, Zygote, Random, ComponentArrays, Optimization, OptimizationOptimJL, Accessors

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
using .PipelineUtils: log_csv, diff3
using .KolmogorovArnoldNets
using .Optimisation: opt_get

mutable struct optim_trainer
    model
    params
    state
    train_data::Tuple{AbstractArray, AbstractArray}
    test_data::Tuple{AbstractArray, AbstractArray}
    opt
    loss_fn
    epoch::Int
    max_epochs::Int
    update_grid_bool::Bool
    verbose::Bool
    log_time::Bool
    x::AbstractArray
    y::AbstractArray
end

function init_optim_trainer(rng::AbstractRNG, model, train_data, test_data, optim_optimiser; loss_fn=nothing, max_epochs=100, update_grid_bool=true, verbose=true, log_time=true)
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
    params, state = Lux.setup(rng, model)
    return optim_trainer(model, params, state, train_data, test_data, optim_optimiser, loss_fn, 0, max_epochs, update_grid_bool, verbose, log_time, train_data...)
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

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)
    x_train, y_train = t.train_data
    x_test, y_test = t.test_data

    # Regularisation
    function reg(ps, st)
        acts_scale = st.act_scale
        
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

        for i in eachindex(t.model.act_fcns)
            coeff_l1 = sum(mean(abs.(ps.act_fcns_ps[Symbol("layer_$i")].coef), dims=2))
            coeff_diff_l1 = sum(mean(abs.(diff3(ps.act_fcns_ps[Symbol("layer_$i")].coef)), dims=2))
            reg_ += (λ_coef * coeff_l1) + (λ_coefdiff * coeff_diff_l1)
        end

        return reg_
    end


    # l1 regularisation loss
    function reg_loss(ps, s)
        ŷ, t.state = t.model(t.x, ps, t.state)
        l2 = mean(sum((ŷ .- t.y).^2))
        reg_ = reg(ps, t.state)
        reg_ = λ * reg_
        return l2 .+ reg_
    end

    if isnothing(t.loss_fn)
        t.loss_fn = reg_loss
    end

    start_time = time()

    function log_callback!(state::Optimization.OptimizationState, obj)
        t.params = state.u

        t.x, t.y = x_test, y_test
        test_loss = t.loss_fn(t.params, nothing)
        reg_ = reg(t.params, t.state)
        t.x, t.y = x_train, y_train

        # Update grid once per epoch if it's time
        new_p = nothing
        if (t.epoch % grid_update_freq == 0) && (t.epoch < stop_grid_update_step) && t.update_grid_bool
            t.model, new_p, t.state = update_grid(t.model, x_train, t.params, t.state)
            @reset state.u = new_p
            t.params = new_p
        end
        
        
        log_csv(t.epoch, time() - start_time, obj, test_loss, reg_, file_name; log_time=t.log_time)
        
        t.epoch += 1

        return false
    end
    
    # Create folders
    !isdir(log_loc) && mkdir(log_loc)
    
    # Create csv with header
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_name = log_loc * "log_" * date_str * ".csv"
    open(file_name, "w") do file
        t.log_time ? write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n") : write(file, "Epoch,Train Loss,Test Loss,Regularisation\n")
    end

    pars = ComponentVector(t.params)
    optf = Optimization.OptimizationFunction(t.loss_fn, Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, pars)

    res = Optimization.solve(optprob, opt_get(t.opt); maxiters=t.max_epochs, callback=log_callback!)
    t.params = res.u
end

end
