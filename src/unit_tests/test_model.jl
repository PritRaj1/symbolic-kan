using Test, Random, Flux, Statistics

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

function test_lock()
    model = KAN([2,5,1]; k=3, grid_interval=5)
    fix_symbolic!(model, 1, 2, 4, "sin", fit_params=false)
    mask1 = model.act_fcns[1].mask
    mask2 = model.symbolic_fcns[1].mask
    @test all(mask1[1, :] .== [1.0, 1.0, 1.0, 1.0, 1.0])
    @test all(mask2[:, 1] .== [1.0, 1.0, 1.0, 1.0, 1.0])
end

function test_opt()
        Random.seed!(123)
        model = KAN([2,5,1]; k=3, grid_interval=5)
        x = randn(100, 2)

        loss(m) = sum((fwd!(m, x) .- 1).^2)
        loss_val, grad = Flux.withgradient(m -> loss(m), model)
        @test abs(loss_val) > 0
end


# test_fwd()
# test_grid()
# test_lock()
test_opt()