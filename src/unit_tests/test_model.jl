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

"""
Example 1 
        ---------
        >>> # when fit_params_bool = False
        >>> model = KAN(width=[2,5,1], grid=5, k=3)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=False)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])
                    
        Example 2
        ---------
        >>> # when fit_params_bool = True
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.normal(0,1,size=(100,2))
        >>> model(x) # obtain activations (otherwise model does not have attributes acts)
        >>> model.fix_symbolic(0,1,3,'sin',fit_params_bool=True)
        >>> print(model.act_fun[0].mask.reshape(2,5))
        >>> print(model.symbolic_fun[0].mask.reshape(2,5))
        r2 is 0.8131332993507385
        r2 is not very high, please double check if you are choosing the correct symbolic function.
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 0., 1., 1.]])
        tensor([[0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0.]])
        '''
        """

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