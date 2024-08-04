using ConfParser, Random

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
using .Utils: round_formula
using .Plotting

FUNCTION = x -> x[1] * x[2]
STRING_VERSION = "x1 * x2"
FILE_NAME = "multiply"

Random.seed!(123)
model = KAN_model([2,5,1]; k=3, grid_interval=5)
ps, st = Lux.setup(Random.default_rng(), model)

train_data, test_data = create_data(FUNCTION, N_var=2, x_range=(-1,1), N_train=100, N_test=100, normalise_input=false, init_seed=1234)
opt = create_optim_opt(model, "bfgs", "backtrack")
trainer = init_optim_trainer(Random.default_rng(), model, train_data, test_data, opt; max_iters=100, verbose=true)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(seed, model, ps, st)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(Random.default_rng(), model, ps, st)
model, ps, st = train!(trainer; λ=1.0, λ_l1=1., λ_entropy=0.1, λ_coef=0.1, λ_coefdiff=0.1, grid_update_num=5, stop_grid_update_step=10)
model, ps, st = prune(Random.default_rng(), model, ps, st)
y, scales, st = model(train_data[1], ps, st)

plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[STRING_VERSION], title="KAN", file_name=FILE_NAME)
model, ps, st = auto_symbolic(model, ps, st; lib=["x", "x^2", "sqrt"])

formula, x0, st = symbolic_formula(model, ps, st)
formula = round_formula(string(formula[1]); digits=1)
println("Formula: ", formula)
plot_kan(model, st; mask=true, in_vars=["x1", "x2"], out_vars=[formula], title="Symbolic KAN", file_name=FILE_NAME*"symbolic")

println("Formula: ", formula)

open("formula.txt", "w") do file
    write(file, formula)
end