using Test

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets

function test_fwd()
    model = KAN([2,5,3]; k=3, grid_interval=5)
    x = randn(100, 2)
    y = model(x)
    @test all(size(y) == (100, 3))
end

test_fwd()