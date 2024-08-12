using ConfParser, Random, Lux, LaTeXStrings

conf = ConfParse("config/pred_multiply_config.ini")
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

FUNCTION = x -> x[:,1] .* x[:,2]
STRING_VERSION = "x1 * x2"
FILE_NAME = "multiply"

### Pipeline hyperparams ###
max_iters = parse(Int, retrieve(conf, "PIPELINE", "max_iters"))
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
g_lims = (grid_lower_lim, grid_upper_lim)

### Optimisation hyperparams ###
type = retrieve(conf, "OPTIMIZER", "type")
linesearch = retrieve(conf, "OPTIMIZER", "linesearch")
m = parse(Int, retrieve(conf, "OPTIMIZER", "m"))
c_1 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_1"))
c_2 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_2"))
ρ = parse(Float64, retrieve(conf, "OPTIMIZER", "ρ"))
α0 = parse(Float64, retrieve(conf, "OPTIMIZER", "init_LR"))

### Schedulers ###
init_noise = parse(Float64, retrieve(conf, "SCHEDULES", "init_stochasticity"))
noise_decay = parse(Float64, retrieve(conf, "SCHEDULES", "stochasticity_decay"))
init_grid_update_freq = parse(Int, retrieve(conf, "SCHEDULES", "init_grid_update_freq"))
grid_update_freq_decay = parse(Float64, retrieve(conf, "SCHEDULES", "grid_update_freq_decay"))

seed = Random.seed!(123)

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

train_data, test_data = create_data(FUNCTION, N_var=2, x_range=lims, N_train=N_train, N_test=N_test, normalise_input=normalise, init_seed=seed)
opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ, init_α=α0)

model = KAN_model([2, 5, 1]; k=k, grid_interval=G, grid_range=g_lims, σ_scale=w_scale, bias_trainable=train_bias, base_act=activation)
ps, st = Lux.setup(seed, model)
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=max_iters, verbose=true, update_grid_bool=true, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff)

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=[STRING_VERSION], title="Trained KAN", file_name=FILE_NAME*"_trained")
model, ps, st = prune(Random.default_rng(), model, ps, st)

trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=max_iters, verbose=true, update_grid_bool=false, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff)

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=["x_1 * x_2"], title="Pruned KAN", file_name=FILE_NAME*"_pruned")
ps, st = cpu_device()(ps), cpu_device()(st)
model, ps, st = auto_symbolic(model, ps, st; lib = ["x", "x^2", "sqrt"])
formula, x0, st = symbolic_formula(model, ps, st)
formula = latexstring(formula[1])
println("Formula: ", formula)

plot_kan(model, st; mask=true, in_vars=["x_1", "x_2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"_symbolic")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end