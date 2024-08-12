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

seed = Random.seed!(123)
model = KAN_model([4,6,2]; k=4, grid_interval=5)
ps, st = Lux.setup(seed, model)

opt = create_optim_opt("bfgs", "strongwolfe")
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=1000, verbose=true)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=1.0, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(seed, model, ps, st; threshold=0.01)
y, st = model(X_test, ps, st)
st = cpu_device()(st)
model, ps, st = auto_symbolic(model, ps, st; lib=["sin", "cos", "exp", "x^2", "x", "exp", "log"])
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=20, verbose=true)
model, ps, st = train!(trainer; ps=ps, st=st, λ=1.0, λ_l1=1., λ_entropy=1.0, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)

formula, x0, st = symbolic_formula(model, ps, st)
formula = round_formula(string(formula[1]); digits=1)
plot_kan(model, st; mask=true, in_vars=["t", "θ1", "ω1", "ω2"], out_vars=["θ1", "θ2"], title="Pruned Double Pendulum KAN", file_name="double_pendulum_kan")

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