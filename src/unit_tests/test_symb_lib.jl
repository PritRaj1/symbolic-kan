using Test

include("../symbolic_lib.jl")
using .SymbolicLib: f_inv, f_inv2, f_inv3, f_inv4, f_inv5, f_sqrt, f_power1d5, f_invsqrt, f_log, f_tan, f_arctanh, f_arcsin, f_arccos, f_exp, SYMBOLIC_LIB

x_test, y_test = [0.1, 0.2], [0.1, 0.2]

# Test singularity protection functions
@test all(f_inv3(x_test, y_test) .≈ ([2.1544346900318834, 1.7099759466766968], [0.004641588833612781, 0.023392141905702935]))
@test all(f_inv4(x_test, y_test) .≈ ([1.7782794100389228, 1.4953487812212205], [0.1, 0.2]))
@test all(f_inv5(x_test, y_test) .≈ ([1.5848931924611134, 1.379729661461215], [0.006309573444801934, 0.028991186547107823]))
@test all(f_sqrt(x_test, y_test) .≈ ([100.0, 25.0], [100.0, 25.0]))
@test all(f_power1d5(x_test, y_test) ≈ [0.0316227766016838, 0.0894427190999916])
@test all(f_invsqrt(x_test, y_test) .≈ ([100.0, 25.0], [0.1, 0.2]))
@test all(f_log(x_test, y_test) .≈ ([0.9048374180359595, 0.8187307530779818], [-0.1, -0.2]))
@test all(f_tan(x_test, y_test) .≈ ([1.4711276743037345, 1.3734007669450157], [0.09997747663138791, 0.19962073122240753]))
@test all(f_arctanh(x_test, y_test) .≈ ([0.9004320053750442, 0.802724679775096], [0.1, 0.2]))
@test all(f_arcsin(x_test, y_test)[2] ≈ [0.1001674211615598, 0.2013579207903308])
@test all(f_arccos(x_test, y_test)[2] ≈ [1.4706289056333368, 1.369438406004566])
@test all(f_exp(x_test, y_test) .≈ ([-2.3025850929940455, -1.6094379124341003], [0.1, 0.2]))

# Test symbolic library
@test SYMBOLIC_LIB["x"][1](x_test) ≈ x_test
@test SYMBOLIC_LIB["x"][2](x_test)[1] ≈ x_test
@test SYMBOLIC_LIB["x^2"][1](x_test) ≈ x_test.^2
@test SYMBOLIC_LIB["x^2"][2](x_test)[1] ≈ x_test.^2
@test SYMBOLIC_LIB["x^3"][1](x_test) ≈ x_test.^3
@test SYMBOLIC_LIB["x^3"][2](x_test)[1] ≈ x_test.^3
@test SYMBOLIC_LIB["x^4"][1](x_test) ≈ x_test.^4
@test SYMBOLIC_LIB["x^4"][2](x_test)[1] ≈ x_test.^4
@test SYMBOLIC_LIB["x^5"][1](x_test) ≈ x_test.^5
@test SYMBOLIC_LIB["x^5"][2](x_test)[1] ≈ x_test.^5
@test SYMBOLIC_LIB["1/x"][1](x_test) ≈ 1 ./ x_test
@test SYMBOLIC_LIB["1/x"][2](x_test)[1] ≈ 1 ./ x_test
@test SYMBOLIC_LIB["1/x^2"][1](x_test) ≈ 1 ./ x_test.^2
@test SYMBOLIC_LIB["1/x^2"][2](x_test)[1] ≈ 1 ./ x_test.^2
@test SYMBOLIC_LIB["1/x^3"][1](x_test) ≈ 1 ./ x_test.^3
@test SYMBOLIC_LIB["1/x^3"][2](x_test)[1] ≈ 1 ./ x_test.^3
@test SYMBOLIC_LIB["1/x^4"][1](x_test) ≈ 1 ./ x_test.^4
@test SYMBOLIC_LIB["1/x^4"][2](x_test)[1] ≈ 1 ./ x_test.^4
@test SYMBOLIC_LIB["1/x^5"][1](x_test) ≈ 1 ./ x_test.^5
@test SYMBOLIC_LIB["1/x^5"][2](x_test)[1] ≈ 1 ./ x_test.^5
@test SYMBOLIC_LIB["sqrt"][1](x_test) ≈ sqrt.(x_test)
@test SYMBOLIC_LIB["sqrt"][2](x_test)[1] ≈ sqrt.(x_test)
@test SYMBOLIC_LIB["x^0.5"][1](x_test) ≈ sqrt.(x_test)
@test SYMBOLIC_LIB["x^0.5"][2](x_test)[1] ≈ sqrt.(x_test)
@test SYMBOLIC_LIB["x^1.5"][1](x_test) ≈ sqrt.(x_test).^3
@test SYMBOLIC_LIB["x^1.5"][2](x_test)[1] ≈ sqrt.(x_test).^3
@test SYMBOLIC_LIB["1/sqrt(x)"][1](x_test) ≈ 1 ./ sqrt.(x_test)
@test SYMBOLIC_LIB["1/sqrt(x)"][2](x_test)[1] ≈ 1 ./ sqrt.(x_test)
@test SYMBOLIC_LIB["1/x^0.5"][1](x_test) ≈ 1 ./ sqrt.(x_test)
@test SYMBOLIC_LIB["1/x^0.5"][2](x_test)[1] ≈ 1 ./ sqrt.(x_test)
@test SYMBOLIC_LIB["exp"][1](x_test) ≈ exp.(x_test)
@test SYMBOLIC_LIB["exp"][2](x_test)[1] ≈ exp.(x_test)
@test SYMBOLIC_LIB["log"][1](x_test) ≈ log.(x_test)
@test SYMBOLIC_LIB["log"][2](x_test)[1] ≈ log.(x_test)
@test SYMBOLIC_LIB["abs"][1](x_test) ≈ abs.(x_test)


