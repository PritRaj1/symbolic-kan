using Test, Random, Flux, Statistics

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

    x = LinRange(-3, 3, 100) |> x -> reshape(x, 100, 1)
    layer = b_spline_layer(1, 1; num_splines=5, degree=3)
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
    z, postacts = symb_fwd(layer, x)

    @test all(size(z) .== (100, 5))
    @test all(size(postacts) .== (100, 5, 3))

    layer = symbolic_kan_layer(10, 10)
    layer = get_symb_subset(layer, [1,10],[2,3,4])

    @test all(size(layer.mask) .== (3, 2))
    @test layer.in_dim == 2
    @test layer.out_dim == 3
end

# Verify optimisation
function test_opt()
    layer = b_spline_layer(3, 5)
    x = randn(100, 3) 
    loss(m) = sum((fwd(m, x)[1] .- 1).^2)
    loss_val, grad = Flux.withgradient(l -> loss(l), layer)
    @test abs(loss_val) > 0

    sym_layer = symbolic_kan_layer(3, 5)
    x = randn(100, 3)
    loss_sym(m) = sum((symb_fwd(m, x)[1] .- 1).^2)
    loss_val, grad = Flux.withgradient(l -> loss_sym(l), sym_layer)
    @test abs(loss_val) > 0
end

test_spline_lyr()
test_symb_lyr()
test_opt()