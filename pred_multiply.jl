using ConfParser, Random, Lux

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

FUNCTION = x -> x[1] * x[2]
STRING_VERSION = "x1 * x2"
FILE_NAME = "multiply"

### Pipeline hyperparams ###
epochs = parse(Int, retrieve(conf, "PIPELINE", "num_epochs"))
N_train = parse(Int, retrieve(conf, "PIPELINE", "N_train"))
N_test = parse(Int, retrieve(conf, "PIPELINE", "N_test"))
num_grid_updates = parse(Int, retrieve(conf, "PIPELINE", "num_grid_updates"))
final_grid_epoch = parse(Int, retrieve(conf, "PIPELINE", "final_grid_epoch"))
normalise = parse(Bool, retrieve(conf, "PIPELINE", "normalise_data"))
lower_lim = parse(Float64, retrieve(conf, "PIPELINE", "input_lower_lim"))
upper_lim = parse(Float64, retrieve(conf, "PIPELINE", "input_upper_lim"))
lims = (lower_lim, upper_lim)

### Architecture hyperparams ###
k = parse(Int, retrieve(conf, "ARCHITECTURE", "k"))
G = parse(Int, retrieve(conf, "ARCHITECTURE", "G"))
λ = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ"))
λ_l1 = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_l1"))
λ_entropy = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_entropy"))
λ_coef = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_coef"))
λ_coefdiff = parse(Float64, retrieve(conf, "ARCHITECTURE", "λ_coefdiff"))

### Optimisation hyperparams ###
type = retrieve(conf, "OPTIMIZER", "type")
linesearch = retrieve(conf, "OPTIMIZER", "linesearch")
m = parse(Int, retrieve(conf, "OPTIMIZER", "m"))
c_1 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_1"))
c_2 = parse(Float64, retrieve(conf, "OPTIMIZER", "c_2"))
ρ = parse(Float64, retrieve(conf, "OPTIMIZER", "ρ"))

seed = Random.seed!(123)

train_data, test_data = create_data(FUNCTION, N_var=2, x_range=lims, N_train=N_train, N_test=N_test, normalise_input=normalise, init_seed=seed)
opt = create_optim_opt(type, linesearch; m=m, c_1=c_1, c_2=c_2, ρ=ρ)

model = KAN_model([2, 3, 3, 1]; k=k, grid_interval=G)
ps, st = Lux.setup(seed, model)
y, scales, st = model(train_data[1], ps, st)
st = cpu_device()(st)

R2, model, ps, st = fix_symbolic(model, ps, st, 1, 1, 1, "x")
R2, model, ps, st = fix_symbolic(model, ps, st, 1, 2, 1, "x")
R2, model, ps, st = fix_symbolic(model, ps, st, 1, 1, 2, "x^2")
st = remove_edge(st, 1, 2, 2)
st = remove_edge(st, 1, 1, 2)
st = remove_edge(st, 1, 1, 3)
R2, model, ps, st = fix_symbolic(model, ps, st, 1, 2, 3, "x^2")
R2, model, ps, st = fix_symbolic(model, ps, st, 2, 1, 1, "x^2")
st = remove_edge(st, 2, 2, 1)
st = remove_edge(st, 2, 3, 1)
st = remove_edge(st, 2, 1, 2)
R2, model, ps, st = fix_symbolic(model, ps, st, 2, 2, 2, "x")
st = remove_edge(st, 2, 3, 2)
st = remove_edge(st, 2, 1, 3)
st = remove_edge(st, 2, 2, 3)
R2, model, ps, st = fix_symbolic(model, ps, st, 2, 3, 3, "x")
R2, model, ps, st = fix_symbolic(model, ps, st, 3, 1, 1, "x")
R2, model, ps, st = fix_symbolic(model, ps, st, 3, 2, 1, "x")
R2, model, ps, st = fix_symbolic(model, ps, st, 3, 3, 1, "x")
plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[STRING_VERSION], title="KAN", file_name=FILE_NAME*"_fixed")

trainer = init_optim_trainer(seed, model, train_data, test_data, opt; max_iters=epochs, verbose=true)
model, ps, st = train!(trainer; ps=ps, st=st, λ=λ, λ_l1=λ_l1, λ_entropy=λ_entropy, λ_coef=λ_coef, λ_coefdiff=λ_coefdiff, grid_update_num=num_grid_updates, stop_grid_update_step=final_grid_epoch)
y, scales, st = model(device(train_data[1]), ps, st)
st = cpu_device()(st)

formula, x0, st = symbolic_formula(model, ps, st)
formula = round_formula(string(formula[1]); digits=1)
println("Formula: ", formula)
plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"_symbolic")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end