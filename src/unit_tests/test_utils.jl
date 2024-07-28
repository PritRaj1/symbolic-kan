using Test, Random

include("../pipeline/utils.jl")
using .PipelineUtils

function test_loaders()
    fcn = x -> x[1] * x[2]
    train_loader, test_loader = create_loaders(fcn)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 2))
    @test all(size(y |> permutedims) .== (32, 1))
    @test all(x[1, :] .- (y[1, :] ./ x[2, :]) .≈ 0.0)

    fcn = x -> x[1] * x[2] * x[3]
    train_loader, test_loader = create_loaders(fcn; N_var=3)
    x, y = first(train_loader)
    @test all(size(x |> permutedims) .== (32, 3))
    @test all(size(y |> permutedims) .== (32, 1))
end

test_loaders()