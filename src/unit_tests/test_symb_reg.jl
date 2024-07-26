include("../pipeline/symbolic_regression.jl")

using .SymbolicRegression

function test_lock()
    model = KAN([2,5,1]; k=3, grid_interval=5)
    fix_symbolic!(model, 1, 2, 4, "sin", fit_params=false)
    mask1 = model.act_fcns[1].mask
    mask2 = model.symbolic_fcns[1].mask
    @test all(mask1[1, :] .== [1.0, 1.0, 1.0, 1.0, 1.0])
    @test all(mask2[:, 1] .== [1.0, 1.0, 1.0, 1.0, 1.0])
end

test_lock()