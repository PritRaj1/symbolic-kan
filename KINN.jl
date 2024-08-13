using MAT, ConfParser, LinearAlgebra, Lux, ComponentArrays

conf = ConfParse("config/pred_function_config.ini")
parse_conf!(conf)

include("src/architecture/kan_model.jl")
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
Boundary = data["Boundary"]

# True displacement
s = data["disp_data"]

# Connectivity matrix (for plotting)
t = data["t"]

# Collocation points
x_full = data["p_full"] # All
x_excl = data["p"] # Excluding boundary

# Material properties
E = 10
μ = 0.3

# Hooke's law for plane stress
stiff = (E / (1 - μ^2)) * [1 μ 0; μ 1 0; 0 0 (1-μ)/2]
stiff = reshape(stiff, (1, size(stiff)...))

# Broadcast stiffness for multiplication later
stiff_bc = stiff
stiff = repeat(stiff, outer=(length(x_excl), 1, 1))
stiff_bc = repeat(stiff_bc, outer=(length(Boundary), 1, 1))

## Models ##
seed = Random.seed!(1234)
stress_KAN = KAN_model([2, 300, 300, 2]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)
disp_KAN = KAN_model([2,260,260,260,260,3]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)

stress_ps, stress_st = Lux.setup(seed, stress_KAN)
disp_ps, disp_st = Lux.setup(seed, disp_KAN)

all_ps = vcat(ComponentVector(stress_ps), ComponentVector(disp_ps))

opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ, init_α=α0)
secondary_opt = create_optim_opt(type_2, linesearch_2; m=m_2, c_1=c_1_2, c_2=c_2_2, ρ=ρ_2, init_α=α0_2)

