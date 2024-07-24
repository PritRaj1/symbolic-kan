using Test, Random, Flux, Zygote, Optim, FluxOptTools, Statistics

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

function test_param_grad()
        Random.seed!(123)
        m = KAN([2,5,1]; k=3, grid_interval=5)
        x = randn(100, 2)
        
        loss() = sum((fwd!(m, x) .- 1).^2)
        pars = Flux.params(m)
        lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
        res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
        println(res.minimizer)
end


# test_fwd()
# test_grid()
# test_lock()
test_param_grad()