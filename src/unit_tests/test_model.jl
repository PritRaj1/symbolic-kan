using Test, Random, Lux, Statistics, Zygote, ComponentArrays

include("../architecture/kan_model.jl")
using .KolmogorovArnoldNets

function test_fwd()
    Random.seed!(123)
    model = KAN_model([2,5,3]; k=3, grid_interval=5)
    ps = Lux.initialparameters(Random.default_rng(), model)
    st = Lux.initialstates(Random.default_rng(), model)

    x = randn(Float32, 100, 2)
    y, _, st = model(x, ps, st)
    @test all(size(y) .== (100, 3))
end

function test_grid()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    ps, st = Lux.setup(Random.default_rng(), model)

    before = model.act_fcns[Symbol("act_lyr_1")].grid[1, :]
    
    x = randn(Float32, 100, 2) .* 5
    model, ps = update_grid(model, x, ps, st)
    
    after = model.act_fcns[Symbol("act_lyr_1")].grid[1, :]
    @test abs(sum(before) - sum(after)) > 0.1
end

function test_opt()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    ps, st = Lux.setup(Random.default_rng(), model)
    
    x = randn(Float32, 100, 2)

    function loss(ps)
        y, _, st = model(x, ps, st)
        return sum((y .- 1).^2)
    end

    pars = ps
    ps = ComponentVector(ps)
    loss_val, grad = Zygote.withgradient(loss, ps)
    grad_2 = Zygote.gradient(p -> loss(p), pars)[1]

    # println(grad[1])
    # println("======================================")
    # println(grad_2)

    grads = gradient(ps) do θ
        loss(Zygote.@showgrad(θ))
    end

    @test abs(loss_val) > 0
end

function test_prune()
    Random.seed!(123)
    model = KAN_model([2,5,1]; k=3, grid_interval=5)
    ps, st = Lux.setup(Random.default_rng(), model)
   
    x = randn(Float32, 100, 2)
    y, st = model(x, ps, st)
    model, ps, st = prune(Random.default_rng(), model, ps, st)
    y, st = model(x, ps, st)
end

# @testset "KAN_model Tests" begin
#     # test_fwd()
#     # test_grid()
#     test_opt()
#     # test_prune()
# end

test_opt()