"""
As a first task, I chose to apply the KAN towards predicting the double pendulum 
because it's quick to implement, and looks cool as a GIF. 

Besides, I'm interested to see whether or not the symbolic KAN can unpack its formuala, 
(probably not). But I only generated data comprising a single swing, and 
the double pendulum is a sequence modelling problem, so a simple FCNN (fully connected neural net)
is not a suitable choice of architecture to do this with :P
"""

using CSV
using DataFrames
using Random
using Statistics
using ConfParser
using Lux
using Plots
using Accessors

conf = ConfParse("config/pred_pendulum_config.ini")
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
    "silu" => x -> x .* NNlib.sigmoid.(x)
)[base_act]

FILE_NAME = "pendulum"

data = CSV.read("data/double_pendulum/double_pendulum_data.csv", DataFrame)
sort!(data, :time)
times = data.time

# Reshape the data: (time_steps, num_variables)
X = Matrix(data[:, [:time, :θ1, :ω1, :ω2]])
y = Matrix(data[:, [:θ1, :θ2]])

X_sorted = copy(X)

println("Data shape: ", size(X), ", ", size(y))

split = parse(Float64, retrieve(conf, "DOUBLE_PENDULUM", "data_split"))
split_idx = floor(Int, split * size(X, 1))
X_train, X_test = X[1:split_idx, :], X[split_idx+1:end, :]
y_train, y_test = y[1:split_idx, :], y[split_idx+1:end, :]
train_data = (X_train, y_train)
test_data = (X_test, y_test)

seed = Random.seed!(1234)

model = KAN_model([4,2,2,2]; k=4, grid_interval=5)
ps, st = Lux.setup(seed, model)
_, _, st = model(train_data[1], ps, st) # warmup for plotting
st = cpu_device()(st)

["t", "θ1", "ω1", "ω2"]
plot_kan(model, st; mask=true, in_vars=["t", "θ1", "ω1", "ω2"], out_vars=["θ1", "θ2"], title="KAN", file_name=FILE_NAME*"_before")

trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")
model, ps, st = prune(seed, model, ps, st)

# After pruning remember to reinit the trainer
trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")
model, ps, st = prune(seed, model, ps, st)

trainer = init_optim_trainer(seed, model, train_data, test_data, opt, secondary_opt; max_iters=max_iters, secondary_iters=max_iters_2, verbose=true, noise=init_noise, noise_decay=noise_decay, grid_update_freq=init_grid_update_freq, grid_update_decay=grid_update_freq_decay, batch_size=batch_size)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")

plot_kan(model, st; mask=true, in_vars=["t", "θ1", "ω1", "ω2"], out_vars=["θ1", "θ2"], title="Pruned KAN", file_name=FILE_NAME*"_trained")

model, ps, st = auto_symbolic(model, ps, st; α_range = (-40, 40), β_range = (-40, 40), lib=["x^2", "cos", "sin", "exp", "tan", "tanh"])
@reset opt.init_α = 1f0
trainer = init_optim_trainer(seed, model, train_data, test_data, opt, nothing; max_iters=10, verbose=true, update_grid_bool=false) # Don't forget to re-init after pruning!
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, img_loc=FILE_NAME*"_training_plots/")

formula, x0, st = symbolic_formula(model, ps, st)
formula = latexstring(formula[1])
println("Formula: ", formula)

plot_kan(model, st; mask=true, in_vars=["t", "θ1", "ω1", "ω2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"_symbolic")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end

function pendulum_positions(ŷ; L1=1.0, L2=1.0)
    x1 = L1 .* sin.(ŷ[:, 1])
    y1 = -L1 .* cos.(ŷ[:, 1])
    x2 = x1 .+ L2 .* sin.(ŷ[:, 2])
    y2 = y1 .- L2 .* cos.(ŷ[:, 2])
    return x1, y1, x2, y2
end

steps_to_plot = parse(Int, retrieve(conf, "DOUBLE_PENDULUM", "num_plot"))
X_gif = X_sorted[1:steps_to_plot, :]
ŷ, st = model(X_gif, ps, st)
ŷ = cpu_device()(ŷ)
x1, y1, x2, y2 = pendulum_positions(ŷ; L1=1.0, L2=1.0)

plot_size = (800, 800)
lim = 2.2
trail_length = 100

anim = @animate for i in 1:length(times)
    plot(size=plot_size, aspect_ratio=:equal, xlim=(-lim, lim), ylim=(-lim, lim),
        legend=false, grid=false, axis=false, title="KAN Predicted Double Pendulum")
    
    # Trail
    trail_start = max(1, i - trail_length)
    plot!(x2[trail_start:i], y2[trail_start:i], color=:lightblue, alpha=0.5, lw=2)
    
    # Pendulum rods
    plot!([0, x1[i], x2[i]], [0, y1[i], y2[i]], color=:black, lw=2)
    
    # Pendulum bobs
    scatter!([0, x1[i], x2[i]], [0, y1[i], y2[i]], 
            color=[:gray, :red, :blue], markersize=[6, 10, 10], 
            markerstrokewidth=0)
end

gif(anim, "figures/pred_double_pendulum.gif", fps=30)

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end