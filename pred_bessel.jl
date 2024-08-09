using ConfParser, Random, Lux, SpecialFunctions

conf = ConfParse("config/config.ini")
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

FUNCTION = x -> besselj0.(20 .* x[:,1]) + x[:,2].^2
STRING_VERSION = "besselj0((20 * x1) + x2^2)"
FILE_NAME = "bessel"

### Pipeline hyperparams ###
epochs = parse(Int, retrieve(conf, "PIPELINE", "num_epochs"))
N_train = parse(Int, retrieve(conf, "PIPELINE", "N_train"))
N_test = parse(Int, retrieve(conf, "PIPELINE", "N_test"))
num_grid_updates = parse(Int, retrieve(conf, "PIPELINE", "num_grid_updates"))
final_grid_epoch = parse(Int, retrieve(conf, "PIPELINE", "final_grid_epoch"))
normalise = parse(Bool, retrieve(conf, "PIPELINE", "normalise_data"))
lower_lim = parse(Float32, retrieve(conf, "PIPELINE", "input_lower_lim"))
upper_lim = parse(Float32, retrieve(conf, "PIPELINE", "input_upper_lim"))
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
g_lims = (grid_lower_lim, grid_upper_lim)

### Optimisation hyperparams ###
type = retrieve(conf, "OPTIMIZER", "type")
linesearch = retrieve(conf, "OPTIMIZER", "linesearch")
m = parse(Int, retrieve(conf, "OPTIMIZER", "m"))
c_1 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_1"))
c_2 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_2"))
ρ = parse(Float64, retrieve(conf, "OPTIMIZER", "ρ"))
α0 = parse(Float64, retrieve(conf, "OPTIMIZER", "α0"))

seed = Random.seed!(123)

train_data, test_data = create_data(FUNCTION, N_var=2, x_range=lims, N_train=N_train, N_test=N_test, normalise_input=normalise, init_seed=seed)
opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ, init_α=α0)

model = KAN_model([2, 5, 1]; k=k, grid_interval=G, grid_range=g_lims)
ps, st = Lux.setup(seed, model)
y, st = model(train_data[1], ps, st) # warmup for plotting
st = cpu_device()(st)

plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[STRING_VERSION], title="KAN", file_name=FILE_NAME*"_before")

trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=epochs, verbose=true)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, grid_update_num=num_grid_updates, stop_grid_update_step=final_grid_epoch)
model, ps, st = prune(seed, model, ps, st)

# After training remember to reinit the trainer
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=epochs, verbose=true)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, grid_update_num=num_grid_updates, stop_grid_update_step=final_grid_epoch)

model, ps, st = auto_symbolic(model, ps, st; α_range = (-40, 40), β_range = (-40, 40))
trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=20, verbose=true) # Don't forget to re-init after pruning!
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, grid_update_num=num_grid_updates, stop_grid_update_step=final_grid_epoch)

_, x0, st, formula = symbolic_formula(model, ps, st)
println("Formula: ", formula)

plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"_after")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end