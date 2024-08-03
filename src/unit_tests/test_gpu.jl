ENV["GPU"] = "true"

using Lux, LuxCUDA, CUDA, KernelAbstractions, Tullio, Test, Random, Zygote

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
include("../architecture/kan_model.jl")
include("../utils.jl")
using .dense_kan
using .symbolic_layer
using .KolmogorovArnoldNets
using .Utils: device

# Test b_spline_layer
function test_spline_lyr()
    x = randn(100, 3) |> device
    l = KAN_Dense(3, 5) 
    ps = Lux.initialparameters(Random.GLOBAL_RNG, l) |> device
    st = Lux.initialstates(Random.GLOBAL_RNG, l) |> device
    y, st = l(x, ps, st)
    l, grads = Zygote.withgradient(p -> sum(l(x, p, st)[1]), ps)
    @test !isnothing(y)
end

# Test symbolic layer
function test_symb_lyr()
    layer = SymbolicDense(3, 5) 
    x = randn(100, 3) |> device
    ps = Lux.initialparameters(Random.GLOBAL_RNG, layer) |> device
    st = Lux.initialstates(Random.GLOBAL_RNG, layer) |> device
    z, st = layer(x, ps, st)
    l, grads = Zygote.withgradient(p -> sum(layer(x, p, st)[1]), ps)
    @test !isnothing(l)
end

function test_model()
    Random.seed!(123)
    model = KAN_model([2,5,3]; k=3, grid_interval=5) 
    ps = Lux.initialparameters(Random.default_rng(), model) |> device
    st = Lux.initialstates(Random.default_rng(), model) |> device

    x = randn(Float32, 100, 2) |> device
    y, _, st = model(x, ps, st)
    l, grads = Zygote.withgradient(p -> sum(model(x, p, st)[1]), ps)
    @test all(size(y) .== (100, 3))
    @test !isnothing(l)
end

function test_grid()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5) 
    ps = Lux.initialparameters(Random.default_rng(), model) |> device
    st = Lux.initialstates(Random.default_rng(), model) |> device

    before = model.act_fcns[1].grid[1, :]
    
    x = randn(Float32, 100, 2) .* 5 |> device
    model, ps, st = update_grid(model, x, ps, st)
    after = model.act_fcns[1].grid[1, :]
    @test abs(sum(before) - sum(after)) > 0.1
end

# test_spline_lyr()
# test_symb_lyr()
# test_model()
# test_grid()