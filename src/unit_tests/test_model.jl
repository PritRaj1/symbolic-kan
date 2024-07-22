using Test, Random

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets

function test_fwd()
    Random.seed!(123)
    model = KAN([2,5,3]; k=3, grid_interval=5)
    Random.seed!(123)
    x = randn(100, 2)
    y = fwd!(model, x)
    @test all(size(y) .== (100, 2))
end

function test_grid()
    Random.seed!(123)
    model = KAN([2,5,1]; k=3, grid_interval=5)
    before = model.act_fcns[1].grid[1, :]
    Random.seed!(123)
    x = randn(100, 2) .* 5
    update_grid!(model, x)
    after = model.act_fcns[1].grid[1, :]
    @test abs(sum(before) - sum(after)) > 0.1
end

test_fwd()
test_grid()