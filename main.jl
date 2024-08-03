using CSV
using DataFrames
using Random
using Statistics
using ConfParser
using Lux
using Plots

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
using .Utils: round_formula

conf = ConfParse("config/data_generation_config.ini")
parse_conf!(conf)

data = CSV.read("data/double_pendulum_data.csv", DataFrame)
sort!(data, :time)
times = data.time

# Reshape the data: (time_steps, num_variables)
X = Matrix(data[:, [:time, :θ1, :ω1, :ω2]])
y = Matrix(data[:, [:θ1, :θ2]])

X_sorted = copy(X)

println("Data shape: ", size(X), ", ", size(y))

split = parse(Float64, retrieve(conf, "DATA", "split"))
split_idx = floor(Int, split * size(X, 1))
X_train, X_test = X[1:split_idx, :], X[split_idx+1:end, :]
y_train, y_test = y[1:split_idx, :], y[split_idx+1:end, :]
train_data = (X_train, y_train)
test_data = (X_test, y_test)

seed = Random.seed!(123)
model = KAN_model([4,5,5,2]; k=4, grid_interval=5)
ps, st = Lux.setup(seed, model)

opt = create_optim_opt(model, "bfgs", "backtrack")
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=4, verbose=true)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(seed, model, ps, st; threshold=0.001)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(Random.default_rng(), model, ps, st; threshold=0.001)
y, scales, st = model(train_data[1], ps, st)
model, ps, st = auto_symbolic(model, ps, st)

formula, x0, st = symbolic_formula(model, ps, st)
formula = round_formula(string(formula[1]); digits=1)
plot_kan(model, st; mask=true, in_vars=["t", "θ1", "ω1", "ω2"], out_vars=["θ1", "θ2"], title="Pruned Double Pendulum KAN", model_name="double_pendulum_kan")

function predict_angular_velocities(model, time, θ1, θ2; ps, st)
    input = [time, θ1, θ2]
    y, scales, st = model(input, ps, st)
    return y
end

function pendulum_positions(ŷ; L1=1.0, L2=1.0)
    x1 = L1 .* sin.(ŷ[:, 1])
    y1 = -L1 .* cos.(ŷ[:, 1])
    x2 = x1 .+ L2 .* sin.(ŷ[:, 2])
    y2 = y1 .- L2 .* cos.(ŷ[:, 2])
    return x1, y1, x2, y2
end

ŷ, scales, st = model(X_sorted, ps, st)
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