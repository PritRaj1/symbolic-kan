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
    println(typeof(ps), " ", typeof(st))
    y, st = l(x, ps, st)
    grads = Zygote.gradient(p -> sum(l(x, p, st)[1]), ps)
end

test_spline_lyr()