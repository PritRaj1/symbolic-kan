ENV["GPU"] = "true"

using Lux, LuxCUDA, CUDA, KernelAbstractions, Tullio, Test, Random, Zygote

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
include("../utils.jl")
using .dense_kan
using .symbolic_layer
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
    layer = SymbolicDense(3, 5) |> device
    x = randn(100, 3) |> device
    ps = Lux.initialparameters(Random.GLOBAL_RNG, layer) |> device
    st = Lux.initialstates(Random.GLOBAL_RNG, layer) |> device
    z, st = layer(x, ps, st)
    l, grads = Zygote.withgradient(p -> sum(layer(x, p, st)[1]), ps)
    @test !isnothing(l)
end

test_spline_lyr()