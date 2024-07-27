using Test, Random
using Plots; pythonplot()

include("../utils.jl")
include("../pipeline/utils.jl")
using .PipelineUtils

function test_loaders()
    fcn = x -> x
    train_loader, test_loader = create_loaders(fcn)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 2))
    @test all(size(y |> permutedims) .== (32, 2))
    @test all(x .- y .== 0.0)

    train_loader, test_loader = create_loaders(fcn; N_var=3)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 3))
    @test all(size(y |> permutedims) .== (32, 3))
    @test all(x .- y .== 0.0)
end

function test_optimisers()
    optimiser, scheduler = create_optimiser("adam"; LR=0.01)
    @test scheduler(1, 0.01) ≈ 0.01
    optimiser, scheduler = create_optimiser("lbfgs"; schedule_LR=true, LR=0.01, step=10, decay=0.1, min_LR=0.001)
    @test scheduler(50, 0.01) ≈ 0.001
end

test_loaders()
test_optimisers()