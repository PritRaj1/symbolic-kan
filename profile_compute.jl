using Profile, Lux, Random, ComponentArrays
using ProfileView: @profview

include("src/architecture/kan_model.jl")
using .KolmogorovArnoldNets

model = KAN_model([2, 5, 1]; k=4, grid_interval=5)
ps, st = Lux.setup(Random.default_rng(), model)
ps = ComponentVector(ps)

x = randn(Float32, 100, 2)

function loss(p, st)
    y, scales, new_st = model(x, p, st)
    return sum(y), new_st
end

@profview loss(ps, st)
Profile.print(IOContext(open("profile_data.txt", "w")), format=:flat)