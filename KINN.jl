using MAT, ConfParser, LinearAlgebra, Lux, ComponentArrays, Optimization

conf = ConfParse("config/pred_function_config.ini")
parse_conf!(conf)

include("src/architecture/kan_model.jl")
include("src/utils.jl")
include("src/pipeline/optimisation.jl")
include("src/architecture/kan_model.jl")
include("src/utils.jl")
include("src/pipeline/plot.jl")
using .PipelineUtils: log_csv
using .KolmogorovArnoldNets
using .Optimisation: opt_get
using .Utils: device
using .Plotting
using .KolmogorovArnoldNets

### Pipeline hyperparams ###
N_train = parse(Int, retrieve(conf, "PIPELINE", "N_train"))
N_test = parse(Int, retrieve(conf, "PIPELINE", "N_test"))
normalise = parse(Bool, retrieve(conf, "PIPELINE", "normalise_data"))
lower_lim = parse(Float32, retrieve(conf, "PIPELINE", "input_lower_lim"))
upper_lim = parse(Float32, retrieve(conf, "PIPELINE", "input_upper_lim"))
train_bias = parse(Bool, retrieve(conf, "PIPELINE", "trainable_bias"))
batch_size = parse(Int, retrieve(conf, "PIPELINE", "batch_size"))
lims = (lower_lim, upper_lim)

### Architecture hyperparams ###
k = parse(Int, retrieve(conf, "ARCHITECTURE", "k"))
G = parse(Int, retrieve(conf, "ARCHITECTURE", "G"))
grid_lower_lim = parse(Float32, retrieve(conf, "ARCHITECTURE", "grid_lower_lim"))
grid_upper_lim = parse(Float32, retrieve(conf, "ARCHITECTURE", "grid_upper_lim"))
λ = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ"))
λ_l1 = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_l1"))
λ_entropy = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_entropy"))
λ_coef = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_coef"))
λ_coefdiff = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_coefdiff"))
w_scale = parse(Float32, retrieve(conf, "ARCHITECTURE", "base_init_scale"))
base_act = retrieve(conf, "ARCHITECTURE", "base_activation")
ENV["sparse_init"] = retrieve(conf, "ARCHITECTURE", "sparse_init")
g_lims = (grid_lower_lim, grid_upper_lim)

### Primiary optimisation hyperparams ###
max_iters = parse(Int, retrieve(conf, "PRIMARY_OPTIMISER", "max_iters"))
type = retrieve(conf, "PRIMARY_OPTIMISER", "type")
linesearch = retrieve(conf, "PRIMARY_OPTIMISER", "linesearch")
m = parse(Int, retrieve(conf, "PRIMARY_OPTIMISER", "m"))
c_1 = parse(Float64, retrieve(conf, "PRIMARY_OPTIMISER", "c_1"))
c_2 = parse(Float64, retrieve(conf, "PRIMARY_OPTIMISER", "c_2"))
ρ = parse(Float64, retrieve(conf, "PRIMARY_OPTIMISER", "ρ"))
α0 = parse(Float64, retrieve(conf, "PRIMARY_OPTIMISER", "init_LR"))

### Secondary optimisation hyperparams ###
max_iters_2 = parse(Int, retrieve(conf, "SECONDARY_OPTIMISER", "max_iters"))
type_2 = retrieve(conf, "SECONDARY_OPTIMISER", "type")
linesearch_2 = retrieve(conf, "SECONDARY_OPTIMISER", "linesearch")
m_2 = parse(Int, retrieve(conf, "SECONDARY_OPTIMISER", "m"))
c_1_2 = parse(Float64, retrieve(conf, "SECONDARY_OPTIMISER", "c_1"))
c_2_2 = parse(Float64, retrieve(conf, "SECONDARY_OPTIMISER", "c_2"))
ρ_2 = parse(Float64, retrieve(conf, "SECONDARY_OPTIMISER", "ρ"))
α0_2 = parse(Float64, retrieve(conf, "SECONDARY_OPTIMISER", "init_LR"))

### Schedulers ###
init_noise = parse(Float64, retrieve(conf, "SCHEDULES", "init_stochasticity"))
noise_decay = parse(Float64, retrieve(conf, "SCHEDULES", "stochasticity_decay"))
init_grid_update_freq = parse(Int, retrieve(conf, "SCHEDULES", "init_grid_update_freq"))
grid_update_freq_decay = parse(Float64, retrieve(conf, "SCHEDULES", "grid_update_freq_decay"))

### Symbolic reg fitting ###
ENV["num_g"] = retrieve(conf, "PARAM_FITTING", "num_g")
ENV["iters"] = retrieve(conf, "PARAM_FITTING", "iters")
ENV["coeff_type"] = retrieve(conf, "PARAM_FITTING", "coeff_type")

activation = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
    "tanh" => NNlib.tanh,
    "silu" => x -> x .* NNlib.sigmoid.(x),
)[base_act]

## Elastic boundary data ##
data = matread("data/elastic_boundary/plate_data.mat")

# BCs
L_boundary = data["L_boundary"]
R_boundary = data["R_boundary"]
T_boundary = data["T_boundary"]
B_boundary = data["B_boundary"]
C_boundary = data["C_boundary"]
Boundary = data["Boundary"]

# True displacement
δ = data["disp_data"]

# Connectivity matrix (for plotting)
t = data["t"]

# Collocation points
x_full = data["p_full"] # All
x_excl = data["p"] # Excluding boundary

# Material properties
E = 1f1 # Young's modulus
μ = 3f-1 # Poisson's ratio
τ_R = 1f-1 # Traction on the right boundary
τ_T = 0f0 # Traction on the top boundary

# Hooke's law for plane stress
stiff = (E / (1 - μ^2)) * [1 μ 0; μ 1 0; 0 0 (1-μ)/2]
stiff = reshape(stiff, (1, size(stiff)...))

# Broadcast stiffness for multiplication later
stiff_bc = stiff
stiff = repeat(stiff, outer=(length(x_excl), 1, 1))
stiff_bc = repeat(stiff_bc, outer=(length(Boundary), 1, 1))

function train(init_params, init_σ_state, init_δ_state, log_loc="logs/", reg_factor=1.0, mag_threshold=1e-16, plot_bool=true, img_loc="training_plots/")
    
    ## Models ##
    seed = Random.seed!(1234)
    stress_KAN = KAN_model([2, 300, 300, 2]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)
    disp_KAN = KAN_model([2,260,260,260,260,3]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)

    σ_ps, σ_st = Lux.setup(seed, stress_KAN)
    δ_ps, δ_st = Lux.setup(seed, disp_KAN)

    num_σ_ps = length(σ_ps)

    all_ps = vcat(ComponentVector(σ_ps), ComponentVector(δ_ps))

    # Data sample
    rand_idx = rand(seed, 1:length(x_full), 50)
    δ_fix = δ[rand_idx, :]

    step, epoch = 1, 1

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
            entropy = -1 * sum(p .* log.(2, p .+ 1f-4))
            reg_ += (l1 * λ_l1) + (entropy * λ_entropy)
        end

        for i in eachindex(t.model.depth)
            coeff_l1 = sum(mean(abs.(ps[Symbol("coef_$i")]), dims=2))
            coeff_diff_l1 = sum(mean(abs.(diff(ps[Symbol("coef_$i")]; dims=3)), dims=2))
            reg_ += (λ_coef * coeff_l1) + (λ_coefdiff * coeff_diff_l1)
        end

        return reg_
    end
    
    function loss_fcn(params, nothing)
        
        ### 1. Constitutive loss ###
        σ_pred, σ_scales, σ_st = stress_KAN(x_excl, params[1:num_σ_ps], σ_st)
        δ_pred, δ_scales, δ_st = disp_KAN(x_excl, params[num_σ_ps+1:end], δ_st)

        u = δ_pred[:, 1] # Horizontal
        v = δ_pred[:, 2] # Vertical

        # Compute derivatives
        dudx = Zygote.jacobian(x -> disp_KAN(x, params[num_σ_ps+1:end], δ_st)[:, 1], x_excl)[1]
        dvdx = Zygote.jacobian(x -> disp_KAN(x, params[num_σ_ps+1:end], δ_st)[:, 2], x_excl)[1]

        # Extract diagonal elements (gradients at each point)
        dudx = [dudx[i,i,:] for i in eachindex(dudx[:,1,1])]
        dvdx = [dvdx[i,i,:] for i in eachindex(dvdx[:,1,1])]

        # Strains
        ε_11 = getindex.(dudx, 1)
        ε_22 = getindex.(dvdx, 2)
        ε_12 = 0.5 .* (getindex.(dudx, 2) .+ getindex.(dvdx, 1))

        ε = hcat(ε_11, ε_22, ε_12)
        ε = reshape(ε, (size(ε,1), 3, 1))

        # Define augment stress
        σ_aug = batched_mul(stiff, ε)
        σ_aug = dropdims(σ_aug, dims=3)

        # Define constitutive loss - forcing the augment stress to be equal to the neural network stress
        loss_cons = mean(sum((σ_aug - σ_pred).^2, dims=1))

        ### 2. Boundary loss ###
        σ_bc, σ_scales, σ_st = stress_KAN(Boundary, params[1:num_σ_ps], σ_st)
        δ_bc, δ_scales, δ_st = disp_KAN(Boundary, params[num_σ_ps+1:end], δ_st)

        u_bc = δ_bc[:, 1] # Horizontal
        v_bc = δ_bc[:, 2] # Vertical

        dudx_bc = Zygote.jacobian(x -> disp_KAN(x, params[num_σ_ps+1:end], δ_st)[:, 1], Boundary)[1]
        dvdx_bc = Zygote.jacobian(x -> disp_KAN(x, params[num_σ_ps+1:end], δ_st)[:, 2], Boundary)[1]

        dudx_bc = [dudx_bc[i,i,:] for i in eachindex(dudx_bc[:,1,1])]
        dvdx_bc = [dvdx_bc[i,i,:] for i in eachindex(dvdx_bc[:,1,1])]

        ε_11_bc = getindex.(dudx_bc, 1)
        ε_22_bc = getindex.(dvdx_bc, 2)
        ε_12_bc = 0.5 .* (getindex.(dudx_bc, 2) .+ getindex.(dvdx_bc, 1))

        ε_bc = hcat(ε_11_bc, ε_22_bc, ε_12_bc)
        ε_bc = reshape(ε_bc, (size(ε_bc,1), 3, 1))

        σ_aug_bc = batched_mul(stiff_bc, ε_bc)
        σ_aug_bc = dropdims(σ_aug_bc, dims=3)

        loss_bc = mean(sum((σ_aug_bc - σ_bc).^2, dims=1))

        ### 3. Equilibrium loss ###
        σ_11 = σ_pred[:, 1]
        σ_22 = σ_pred[:, 2]
        σ_12 = σ_pred[:, 3]

        # Enforce equilibrium in x and y planes
        dσ_11dx = Zygote.jacobian(x -> stress_KAN(x, params[1:num_σ_ps], σ_st)[:, 1], x_excl)[1]
        dσ_22dx = Zygote.jacobian(x -> stress_KAN(x, params[1:num_σ_ps], σ_st)[:, 2], x_excl)[1]
        dσ_12dx = Zygote.jacobian(x -> stress_KAN(x, params[1:num_σ_ps], σ_st)[:, 3], x_excl)[1]

        eq_x1 = dσ_11dx[:,1] + dσ_12dx[:,2]
        eq_x2 = dσ_12dx[:,1] + dσ_22dx[:,2]

        # Enforce equilibrium - zero body forces
        loss_eq = mean(sum((eq_x1 .- 0f0).^2, dims=1)) + mean(sum((eq_x2 .- 0f0).^2, dims=1))

        ### 4. Boundary conditions ###
        u_L, δ_scales, δ_st  = disp_KAN(L_boundary, params[num_σ_ps+1:end], δ_st)
        u_B, δ_scales, δ_st  = disp_KAN(B_boundary, params[num_σ_ps+1:end], δ_st)

        σ_R, σ_scales, σ_st = stress_KAN(R_boundary, params[1:num_σ_ps], σ_st)
        σ_T, σ_scales, σ_st = stress_KAN(T_boundary, params[1:num_σ_ps], σ_st)
        σ_C, σ_scales, σ_st = stress_KAN(C_boundary, params[1:num_σ_ps], σ_st)

        loss_BC_L = mean(sum((u_L[:,1] .- 0f0).^2, dims=1)) 
        loss_BC_B = mean(sum((u_B[:,2] .- 0f0).^2, dims=1))
        loss_BC_R = mean(sum((σ_R[:,1] .- τ_R).^2, dims=1)) + mean(sum((σ_R[:,2] .- 0f0).^2, dims=1))
        loss_BC_T = mean(sum((σ_T[:,2] .- τ_T).^2, dims=1)) + mean(sum((σ_T[:,3] .- 0f0).^2, dims=1))
        loss_BC_C = mean(sum(((σ_C[:,1]*C_boundary[:,1] + σ_C[:,3]*C_boundary[:,2]) .- 0f0).^2, dims=1))

        ### 5. Data loss ###
        x_fix = x_full[rand_idx, :]
        u_fix, δ_scales, δ_st = disp_KAN(x_fix, params[num_σ_ps+1:end], δ_st)
        loss_fix = mean(sum((u_fix - δ_fix).^2, dims=1))

        ### 6. Regularisation ###
        σ_reg = reg(params[1:num_σ_ps], σ_scales)
        δ_reg = reg(params[num_σ_ps+1:end], δ_scales)

        return loss_cons + loss_bc + loss_eq + loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C + loss_fix + (σ_reg + δ_reg)*λ
    end

    function grad_fcn(params, nothing)

        # Update grid once per epoch if it's time
        if  (grid_update_freq > 0 && step % grid_update_freq == 0) || step == 1
            stress_KAN, σ_ps = update_grid!(stress_KAN, params[1:num_σ_ps], σ_ps)
            disp_KAN, δ_ps = update_grid!(disp_KAN, params[num_σ_ps+1:end], δ_ps)
            grid_update_freq = floor(grid_update_freq * (2 - grid_update_decay)^step)
        end

        step += 1

        return Zygote.gradient(params -> loss_fcn(params, nothing), params)[1]
    end

    # Callback function for logging
    start_time = time()
    function log_callback!(state, obj)
        reg_ = reg(state.u, scales)
        log_csv(t.epoch, time() - start_time, obj, 0f0, reg_, file_name)
        epoch = epoch + 1
        return false
    end

    opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ, init_α=α0)
    secondary_opt = create_optim_opt(type_2, linesearch_2; m=m_2, c_1=c_1_2, c_2=c_2_2, ρ=ρ_2, init_α=α0_2)

    optf = Optimization.OptimizationFunction(loss_fcn; grad = grad_fcn)
    optprob = Optimization.OptimizationProblem(optf, all_ps)
    res = Optimization.solve(optprob, opt_get(t.opt); 
    maxiters=max_iters, callback=log_callback!, abstol=0f0, reltol=0f0, allow_f_increases=true, allow_outer_f_increases=true, x_tol=0f0, x_abstol=0f0, x_reltol=0f0, f_tol=0f0, f_abstol=0f0, f_reltol=0f0, g_tol=0f0, g_abstol=0f0, g_reltol=0f0,
    outer_x_abstol=0f0, outer_x_reltol=0f0, outer_f_abstol=0f0, outer_f_reltol=0f0, outer_g_abstol=0f0, outer_g_reltol=0f0, successive_f_tol=max_iters)
end

train(all_ps, σ_st, δ_st)


