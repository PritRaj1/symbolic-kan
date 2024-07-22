include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")

using Test
using .dense_kan
using .symbolic_layer

# Test b_spline_layer
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

# Test symbolic_dense
layer = symbolic_dense(3, 5)
z, postacts = layer(x)

@test all(size(z) .== (100, 5))
@test all(size(postacts) .== (100, 5, 3))