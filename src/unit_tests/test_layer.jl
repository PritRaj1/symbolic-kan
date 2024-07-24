using Test, Random, Flux, Zygote, Optim, FluxOptTools, Statistics

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
using .dense_kan
using .symbolic_layer

# Test b_spline_layer
function test_spline_lyr()
    layer = b_spline_layer(3, 5)
    x = randn(100, 3) 
    y = randn(100, 3) 
    z, preacts, postacts, postspline = fwd(layer, x)

    @test all(size(z) .== (100, 5))
    @test all(size(preacts) .== (100, 5, 3))
    @test all(size(postacts) .== (100, 5, 3))
    @test all(size(postspline) .== (100, 5, 3))
    @test all(size(layer.grid) .== (3, 12))

    update_lyr_grid!(layer, x)

    @test all(size(layer.grid) .== (3, 12))

    layer = b_spline_layer(10, 10)
    layer = get_subset(layer, [1,10],[2,3,4])

    @test all(size(layer.mask) .== (2, 3))
    @test layer.in_dim == 2
    @test layer.out_dim == 3
    @test all(size(layer.grid) .== (2, 12))
end

# Test symbolic layer
function test_symb_lyr()
    layer = symbolic_kan_layer(3, 5)
    x = randn(100, 3) 
    z, postacts = layer(x)

    @test all(size(z) .== (100, 5))
    @test all(size(postacts) .== (100, 5, 3))

    layer = symbolic_kan_layer(10, 10)
    layer = get_symb_subset(layer, [1,10],[2,3,4])

    @test all(size(layer.mask) .== (3, 2))
    @test layer.in_dim == 2
    @test layer.out_dim == 3

    layer = symbolic_kan_layer(3, 2)
    lock_symbolic!(layer, 3, 2, "sin")

    @test layer.fcns[2][3](2.4) == sin(2.4)
    @test layer.fcn_names[2][3] == "sin"
    @test all(layer.affine[2, 3, :] == [1.0, 0.0, 1.0, 0.0])

    layer = symbolic_kan_layer(3, 2)
    num = 100
    x = range(-1, 1, length=num) |> collect
    noises = randn(num) .* 0.02
    y = 2 .* x .+ 1 .+ noises
    fcn = "x"
    R2 = lock_symbolic!(layer, 3, 2, fcn; x, y, random=true, seed=123)

    @test layer.fcns[2][3](2.4) == 2.4
    @test layer.fcn_names[2][3] == "x"
    @test layer.affine[2, 3, 1] - 2 < 0.01
    @test layer.affine[2, 3, 2] - 1 < 0.01
    @test layer.affine[2, 3, 3] - 1 < 0.01
    @test layer.affine[2, 3, 4] - 0 < 0.01
    @test R2 >= 0.9
end

function test_param_grad()
    layer = b_spline_layer(3, 5)
    x = randn(100, 3) 

    loss() = sum((fwd(layer, x)[1] .- 1).^2)
    
    params = Flux.params(layer)
    lossfun, gradfun, fg!, p0 = optfuns(loss, params)
    res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))
    println(res.minimizer)
end

# test_spline_lyr()
# test_symb_lyr()
test_param_grad()