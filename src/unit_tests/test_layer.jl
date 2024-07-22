using Test

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
using .dense_kan
using .symbolic_layer

# Test b_spline_layer
function test_spline_lyr()
    layer = b_spline_layer(3, 5)
    x = randn(100, 3) 
    y = randn(100, 3) 
    z, preacts, postacts, postspline = layer(x)

    @test all(size(z) .== (100, 5))
    @test all(size(preacts) .== (100, 5, 3))
    @test all(size(postacts) .== (100, 5, 3))
    @test all(size(postspline) .== (100, 5, 3))
    @test all(size(layer.grid) .== (3, 12))

    update_grid!(layer, x)

    @test all(size(layer.grid) .== (3, 12))
end

# Test symbolic layer
function test_symb_lyr()
    layer = symbolic_kan_layer(3, 5)
    x = randn(100, 3) 
    z, postacts = layer(x)

    @test all(size(z) .== (100, 5))
    @test all(size(postacts) .== (100, 5, 3))

    layer = symbolic_kan_layer(10, 10)
    layer = get_subset(layer, [1,10],[2,3,4])

    @test all(size(layer.mask) .== (3, 2))
    @test layer.in_dim == 2
    @test layer.out_dim == 3
end

test_spline_lyr()
test_symb_lyr()