module OptimTrainer

export init_optim_trainer, train!

using Lux, LuxCUDA, ProgressBars, Dates, Tullio, CSV, Statistics, Zygote, Random, ComponentArrays, Optimization, OptimizationOptimJL, Accessors, ComponentArrays
using NNlib: sigmoid

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
include("../utils.jl")
using .PipelineUtils: log_csv
using .KolmogorovArnoldNets
using .Optimisation: opt_get
using .Utils: device

mutable struct optim_trainer
    model
    params
    state
    train_data::Tuple{AbstractArray, AbstractArray}
    test_data::Tuple{AbstractArray, AbstractArray}
    opt
    loss_fn
    epoch::Int
    max_iters::Int
    update_grid_bool::Bool
    verbose::Bool
    log_time::Bool
    x::AbstractArray
    y::AbstractArray
end

function init_optim_trainer(rng::AbstractRNG, model, train_data, test_data, optim_optimiser; loss_fn=nothing, max_iters=1e5, update_grid_bool=true, verbose=true, log_time=true)
    """
    Initialise trainer for training symbolic model.

    Args:
    - rng: random number generator.
    - model: symbolic model to train.
    - train_data: tuple of training data.
    - test_data: tuple of testing data.
    - optimiser: optimiser object.
    - loss_fn: loss function.
    - max_iters: maximum number of epochs.
    - update_grid_bool: whether to update grid.
    - verbose: whether to print training progress.
    - log_time: whether to log time.

    Returns:
    - t: trainer object.
    """
    params, state = Lux.setup(rng, model)
    params = device(params)
    state = device(state)
    x, y = train_data
    x = device(x)
    y = device(y)
    return optim_trainer(model, params, state, train_data, test_data, optim_optimiser, loss_fn, 0, max_iters, update_grid_bool, verbose, log_time, x, y)
end

function train!(t::optim_trainer; ps=nothing, st=nothing, log_loc="logs/", grid_update_num=5, stop_grid_update_step=10, reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0)
    """
    Train symbolic model.

    Args:
    - t: trainer object.
    - log_loc: location to save logs.
    - grid_update_num: number of times to update grid.
    - stop_grid_update_step: number of epochs to stop updating grid.
    - reg_factor: regularisation factor for non_linear.
    - mag_threshold: threshold for regularisation.
    - λ: regularisation factor.
    - λ_l1: l1 regularisation factor.
    - λ_entropy: entropy regularisation factor.
    - λ_coef: coefficient regularisation factor.
    - λ_coefdiff: coefficient difference regularisation factor.

    Returns:
    - nothing: the trainer object is updated in place.
    """
    λ = Float32(λ)
    λ_l1 = Float32(λ_l1)
    λ_entropy = Float32(λ_entropy)
    λ_coef = Float32(λ_coef)
    λ_coefdiff = Float32(λ_coefdiff)

    grid_update_freq = fld(stop_grid_update_step, grid_update_num)
    x_train, y_train = t.train_data
    x_test, y_test = t.test_data
    
    x_train = device(x_train)
    y_train = device(y_train)
    x_test = device(x_test)
    y_test = device(y_test)

    if !isnothing(ps)
        t.params = device(ps)
    end
    if !isnothing(st)
        t.state = device(st)
    end

    # Regularisation
    function reg(ps, st)
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            # term1 = ifelse.(x .< th, Float32(1), Float32(0))
            # term2 = ifelse.(x .>= th, Float32(1), Float32(0))
            term1 = sigmoid(x .- th)
            term2 = sigmoid(th .- x)
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
        end

        reg_ = Float32(0.0)
        for i in 1:t.model.depth
            vec = reshape(st[Symbol("act_scale_$i")], :)
            p = vec ./ sum(vec)
            l1 = sum(non_linear(vec))
            entropy = -1 * sum(p .* log.(p .+ Float32(1e-2)))
            reg_ += (l1 * λ_l1) + (entropy * λ_entropy)
        end

        for i in eachindex(t.model.act_fcns)
            coeff_l1 = sum(mean(abs.(ps[Symbol("coef_$i")]), dims=2))
            coeff_diff_l1 = sum(mean(abs.(diff(ps[Symbol("coef_$i")]; dims=3)), dims=2))
            reg_ += (λ_coef * coeff_l1) + (λ_coefdiff * coeff_diff_l1)
        end

        return reg_
    end


    # l1 regularisation loss
    function reg_loss(ps, s)
        ŷ, t.state = t.model(t.x, ps, t.state)
        l2 = mean((ŷ - t.y).^2) 
        reg_ = reg(ps, t.state)
        reg_ = λ * reg_
        return l2 + reg_
    end

    if isnothing(t.loss_fn)
        t.loss_fn = reg_loss
    end

    start_time = time()

    function log_callback!(state::Optimization.OptimizationState, obj)
        t.params = state.u 

        if any(isnan.(state.grad))
            println("NaN in gradients")
            grads = state.grad 
            grads = cpu_device()(grads)
            for k in keys(grads)
                if any(isnan.(grads[k]))
                    println("NaN in $k")
                end
            end
        end

        t.x, t.y = x_test, y_test
        test_loss = t.loss_fn(state.u, nothing)
        ŷ, t.state = t.model(t.x, t.params, t.state)
        reg_ = reg(t.params, t.state)
        t.x, t.y = x_train, y_train

        # Update grid once per epoch if it's time
        new_p = nothing
        if (t.epoch % grid_update_freq == 0) && (t.epoch < stop_grid_update_step) && t.update_grid_bool
            t.model, new_p = update_grid(t.model, x_train, t.params, t.state)
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
    println("Created log at $file_name")

    pars = t.params |> ComponentArray
    optf = Optimization.OptimizationFunction(t.loss_fn, Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, pars)
    res = Optimization.solve(optprob, opt_get(t.opt); maxiters=t.max_iters, callback=log_callback!, x_tol=1e-32, f_tol=1e-9, g_tol=1e-12, allow_f_increases=true, allow_outer_f_increases=true, abstol=1e-32, reltol=1e-32)
    t.params = res.minimizer
    return t.model, cpu_device()(t.params), cpu_device()(t.state)
end

end
