using ConfParser, Random, Lux, SpecialFunctions, LaTeXStrings, Accessors

conf = ConfParse("config/pred_function_config.ini")
parse_conf!(conf)

use_gpu = parse(Bool, retrieve(conf, "CUDA", "use_gpu"))

ENV["GPU"] = use_gpu ? "true" : "false"

include("src/pipeline/symbolic_regression.jl")
include("src/architecture/kan_model.jl")
include("src/pipeline/utils.jl")
include("src/pipeline/optim_trainer.jl")
include("src/pipeline/optimisation.jl")
include("src/pipeline/plot.jl")
include("src/utils.jl")
using .KolmogorovArnoldNets
using .SymbolicRegression
using .PipelineUtils
using .OptimTrainer
using .Optimisation
using .Utils: round_formula, device
using .Plotting

FUNCTION = x -> sin.(π .* x[:, 1] + x[:, 2].^2)
STRING_VERSION = "sin(π x_1 + x_2^2)"
FILE_NAME = "sine"

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
init_noise = parse(Float32, retrieve(conf, "SCHEDULES", "init_stochasticity"))
noise_decay = parse(Float32, retrieve(conf, "SCHEDULES", "stochasticity_decay"))
init_grid_update_freq = parse(Int, retrieve(conf, "SCHEDULES", "init_grid_update_freq"))
grid_update_freq_decay = parse(Float32, retrieve(conf, "SCHEDULES", "grid_update_freq_decay"))

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
    "silu" => x -> x .* NNlib.sigmoid.(x)
)[base_act]

seed = Random.seed!(1234)

train_data, test_data = create_data(FUNCTION, N_var=2, x_range=lims, N_train=N_train, N_test=N_test, normalise_input=normalise, init_seed=seed)
opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ, init_α=α0)
secondary_opt = create_optim_opt(type_2, linesearch_2; m=m_2, c_1=c_1_2, c_2=c_2_2, ρ=ρ_2, init_α=α0_2)

model = KAN_model([2, 5, 1]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)
ps, st = Lux.setup(seed, model)
_, _, st = model(train_data[1], ps, st) # warmup for plotting
st = cpu_device()(st)

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=[STRING_VERSION], title="KAN", file_name=FILE_NAME*"_before")

trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")
model, ps, st = prune(seed, model, ps, st)

# After pruning remember to reinit the trainer
trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")
model, ps, st = prune(seed, model, ps, st)

trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=[STRING_VERSION], title="Pruned KAN", file_name=FILE_NAME*"_trained")

model, ps, st = auto_symbolic(model, ps, st; α_range = (-40, 40), β_range = (-40, 40), lib=lib=["x", "x^2", "sin", "exp"])
@reset opt.init_α = 1f0
trainer = init_optim_trainer(seed, model, train_data, test_data, opt, nothing; max_iters=10, verbose=true, update_grid_bool=false) # Don't forget to re-init after pruning!
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")

formula, x0, st = symbolic_formula(model, ps, st)
formula = latexstring(formula[1])
println("Formula: ", formula)

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"_symbolic")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end