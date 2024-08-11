using Test, Random, Lux, Statistics, Zygote

include("../architecture/kan_layer.jl")
include("../architecture/symbolic_layer.jl")
using .dense_kan
using .symbolic_layer

# Test b_spline_layer
function test_spline_lyr()
    x = randn(100, 3)
    l = KAN_Dense(3, 5)
    ps = Lux.initialparameters(Random.GLOBAL_RNG, l)
    st = Lux.initialstates(Random.GLOBAL_RNG, l)
    y, st = l(x, ps, st.mask)
    grads = Zygote.gradient(p -> sum(l(x, p, st.mask)[1]), ps)
    
    preacts, postacts, postspline = st.pre_acts, st.post_acts, st.post_spline

    @test all(size(y) .== (100, 5))
    @test all(size(preacts) .== (100, 5, 3))
    @test all(size(postacts) .== (100, 5, 3))
    @test all(size(postspline) .== (100, 5, 3))
    @test all(size(l.grid) .== (3, 12))

    x = LinRange(-3, 3, 100) |> x -> reshape(x, 100, 1)
    l = KAN_Dense(1, 1; num_splines=5, degree=3)
    ps = Lux.initialparameters(Random.GLOBAL_RNG, l)
    st = Lux.initialstates(Random.GLOBAL_RNG, l)
    grid, coef = update_lyr_grid(l, ps.coef, x)

    @test all(size(grid) .== (1, 12))

    l =  KAN_Dense(10, 10)
    ps = Lux.initialparameters(Random.GLOBAL_RNG, l)
    st = Lux.initialstates(Random.GLOBAL_RNG, l)
    l, ps, new_mask = get_subset(l, ps, st.mask, [1,10],[2,3,4])
    @test all(size(new_mask) .== (2, 3))
    @test l.in_dim == 2
    @test l.out_dim == 3
    @test all(size(l.grid) .== (2, 12))
end

# Test symbolic layer
function test_symb_lyr()
    layer = SymbolicDense(3, 5)
    x = randn(100, 3) 
    ps = Lux.initialparameters(Random.GLOBAL_RNG, layer)
    st = Lux.initialstates(Random.GLOBAL_RNG, layer)
    z, st = layer(x, ps, st.mask)
    grads = Zygote.gradient(p -> sum(layer(x, p, st.mask)[1]), ps)

    @test all(size(z) .== (100, 5))
    @test all(size(st.post_acts) .== (100, 5, 3))

    layer = SymbolicDense(10, 10)
    ps = Lux.initialparameters(Random.GLOBAL_RNG, layer)
    st = Lux.initialstates(Random.GLOBAL_RNG, layer)
    layer, ps, new_mask = get_symb_subset(layer,ps, st.mask, [1,10],[2,3,4])

    @test all(size(new_mask) .== (3, 2))
    @test layer.in_dim == 2
    @test layer.out_dim == 3
end

@testset "KAN_layer Tests" begin
    test_spline_lyr()
    # test_symb_lyr()
end