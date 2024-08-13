module OptimTrainer

export init_optim_trainer, train!

using Lux, LuxCUDA, ProgressBars, Dates, Tullio, CSV, Statistics, Zygote, Random, ComponentArrays, Optimization, OptimizationOptimJL, Accessors, ComponentArrays, Random
using NNlib: sigmoid
using Flux: DataLoader

include("utils.jl")
include("../pipeline/optimisation.jl")
include("../architecture/kan_model.jl")
include("../utils.jl")
include("../pipeline/plot.jl")
using .PipelineUtils: log_csv
using .KolmogorovArnoldNets
using .Optimisation: opt_get
using .Utils: device
using .Plotting

mutable struct optim_trainer
    model
    params
    state
    train_data::Tuple{AbstractArray{Float32}, AbstractArray{Float32}}
    test_data::Tuple{AbstractArray{Float32}, AbstractArray{Float32}}
    b_size::Int
    opt
    secondary_opt
    loss_fn
    epoch::Int
    max_iters::Int
    secondary_iters::Int
    update_grid_bool::Bool
    verbose::Bool
    log_time::Bool
    x::AbstractArray{Float32}
    y::AbstractArray{Float32}
    ε::Float32
    ε_decay::Float32
    grid_update_freq::Int
    grid_update_decay::Float32
    seed::Int
end

function init_optim_trainer(rng::AbstractRNG, model, train_data, test_data, optim_optimiser, secondary_optimiser; batch_size=nothing, loss_fn=nothing, max_iters=1e2, secondary_iters=20, noise=0f0, noise_decay=1f0, grid_update_freq=5, grid_update_decay=1f0, update_grid_bool=true, verbose=true, log_time=true)
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
    batch_size = isnothing(batch_size) ? size(x, 1) : batch_size

    return optim_trainer(model, params, state, train_data, test_data, batch_size, optim_optimiser, secondary_optimiser, loss_fn, 0, max_iters, secondary_iters, update_grid_bool, verbose, log_time, x, y, noise, noise_decay, grid_update_freq, grid_update_decay, 1)
end

function train!(t::optim_trainer; ps=nothing, st=nothing, log_loc="logs/", reg_factor=1.0, mag_threshold=1e-16, 
    λ=0.0, λ_l1=1.0, λ_entropy=0.0, λ_coef=0.0, λ_coefdiff=0.0, plot_bool=true, img_loc="training_plots/")
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
    reg_factor = Float32(reg_factor)
    mag_threshold = Float32(mag_threshold)

    x_train, y_train = t.train_data
    x_test, y_test = t.test_data
    
    x_test = device(x_test)
    y_test = device(y_test)

    rng = Random.seed!(t.seed)
    t.seed += 1
    train_loader = DataLoader((permutedims(x_train), permutedims(y_train)); batchsize=t.b_size, shuffle=true, rng=rng)

    step = 1

    if !isnothing(ps)
        t.params = device(ps)
    end
    if !isnothing(st)
        t.state = device(st)
    end

    # Regularisation
    function reg(ps, scales)
        
        # L2 regularisation
        function non_linear(x; th=mag_threshold, factor=reg_factor)
            term1 = ifelse.(x .< th, 1f0, 0f0)
            term2 = ifelse.(x .>= th, 1f0, 0f0)
            return term1 .* x .* factor .+ term2 .* (x .+ (factor - 1) .* th)
            # s = sigmoid(x .- th)
            # return x .+ s .* ((factor - 1) .* (1 .- s))
        end

        reg_ = 0f0
        for i in 1:t.model.depth
            vec = scales[i, 1:t.model.widths[i]*t.model.widths[i+1]]
            p = vec ./ sum(vec)
            l1 = sum(non_linear(vec))
            entropy = -1 * sum(p .* log.(p .+ 1f-4))
            reg_ += (l1 * λ_l1) + (entropy * λ_entropy)
        end

        for i in eachindex(t.model.depth)
            coeff_l1 = sum(mean(abs.(ps[Symbol("coef_$i")]), dims=2))
            coeff_diff_l1 = sum(mean(abs.(diff(ps[Symbol("coef_$i")]; dims=3)), dims=2))
            reg_ += (λ_coef * coeff_l1) + (λ_coefdiff * coeff_diff_l1)
        end

        return reg_
    end

    if isnothing(t.loss_fn)
        t.loss_fn = (pred, real) -> mean(sum((pred - real).^2; dims=2))
    end

    # l1 regularisation loss
    function reg_loss(ps, s)
        ŷ, scales, st = t.model(t.x, ps, t.state)
        l2 = t.loss_fn(ŷ, t.y)
        reg_ = reg(ps, scales)
        reg_ = λ * reg_

        t.state = st

        return l2 + reg_
    end

    # Single train step
    function grad_fcn(G, u, p)

        t.params = u

        # Update grid once per epoch if it's time
        if  ((t.grid_update_freq > 0 && step % t.grid_update_freq == 0) || step == 1) && t.update_grid_bool
            
            if t.verbose
                println("Updating grid at epoch $(t.epoch), step $step")
            end

            new_model, new_p = update_grid(t.model, device(x_train), u, st)
            t.params = new_p
            t.model = new_model

            t.grid_update_freq = floor(t.grid_update_freq * (2 - t.grid_update_decay)^step)
        end

        step += 1

        grads = zeros(Float32, length(u)) |> device
        for (x, y) in train_loader
            t.x = x |> permutedims |> device
            t.y = y |> permutedims |> device
            grads += Zygote.gradient(θ -> reg_loss(θ, nothing), t.params)[1]
        end

        rng = Random.seed!(t.seed)
        t.seed += 1
        noises = randn(rng, Float32, length(grads)) .* t.ε |> device
        grads = grads .+ noises

        if t.verbose
            println("Epoch $(t.epoch): Loss: $(reg_loss(u, nothing)), Grid updates every: $(t.grid_update_freq), ε: $(t.ε)")
        end

        copy!(G, grads)
        return grads
    end

    start_time = time()

    # Callback function for logging
    function log_callback!(state::Optimization.OptimizationState, obj)
        t.params = state.u
        t.update_grid_bool = true

        # Update stochasticity
        t.ε = t.ε > 0f0 ? t.ε .* t.ε_decay : 0f0

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

        ŷ, scales, st = t.model(t.x, state.u, t.state)
        train_loss = t.loss_fn(ŷ, t.y)

        t.x = x_test
        t.y = y_test
        ŷ, scales, st = t.model(t.x, state.u, t.state)
        test_loss = t.loss_fn(ŷ, t.y)

        reg_ = reg(state.u, scales)

        t.params = state.u
        t.state = st
 
        log_csv(t.epoch, time() - start_time, train_loss, test_loss, reg_, file_name; log_time=t.log_time)

        if plot_bool
            plot_kan(t.model, st; mask=true, title="Epoch $(t.epoch)", folder=img_loc, file_name="epoch_$(t.epoch)")
        end

        t.epoch = t.epoch + 1

        return false
    end
    
    # Create folders
    !isdir(log_loc) && mkdir(log_loc)
    plot_bool && !isdir(img_loc) && mkdir(img_loc)
    
    # Create csv with header
    date_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    file_name = log_loc * "log_" * date_str * ".csv"
    open(file_name, "w") do file
        t.log_time ? write(file, "Epoch,Time (s),Train Loss,Test Loss,Regularisation\n") : write(file, "Epoch,Train Loss,Test Loss,Regularisation\n")
    end
    println("Created log at $file_name")

    # Problem setup - see SciML docs
    pars = ComponentVector(t.params)
    optf = Optimization.OptimizationFunction(reg_loss; grad=grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, pars)
    res = Optimization.solve(optprob, opt_get(t.opt); 
    maxiters=t.max_iters, callback=log_callback!, abstol=0f0, reltol=0f0, allow_f_increases=true, allow_outer_f_increases=true, x_tol=0f0, x_abstol=0f0, x_reltol=0f0, f_tol=0f0, f_abstol=0f0, f_reltol=0f0, g_tol=0f0, g_abstol=0f0, g_reltol=0f0,
    outer_x_abstol=0f0, outer_x_reltol=0f0, outer_f_abstol=0f0, outer_f_reltol=0f0, outer_g_abstol=0f0, outer_g_reltol=0f0, successive_f_tol=t.max_iters)
    
    if !isnothing(t.secondary_opt) && t.secondary_opt.type != "nothing"

        if t.verbose
            println("Starting fine-tuning with $(t.secondary_opt.type)")
        end

        optprob = remake(optprob; u0=res.minimizer)
        res = Optimization.solve(optprob, opt_get(t.secondary_opt);
        maxiters=t.secondary_iters, callback=log_callback!, abstol=0f0, reltol=0f0, allow_f_increases=true, allow_outer_f_increases=true, x_tol=0f0, x_abstol=0f0, x_reltol=0f0, f_tol=0f0, f_abstol=0f0, f_reltol=0f0, g_tol=0f0, g_abstol=0f0, g_reltol=0f0,
        outer_x_abstol=0f0, outer_x_reltol=0f0, outer_f_abstol=0f0, outer_f_reltol=0f0, outer_g_abstol=0f0, outer_g_reltol=0f0, successive_f_tol=t.max_iters)
    end

    t.params = res.minimizer
    return t.model, cpu_device()(res.minimizer), cpu_device()(t.state)
end

end
