module SymbolicLib

using Flux, LinearAlgebra, SymPy

# Helper functions
nan_to_num(x) = replace(x, NaN => 0.0, Inf => 0.0, -Inf => 0.0)
sign(x) = x < 0 ? -1 : (x > 0 ? 1 : 0)
safe_log(x) = x > 0 ? log(x) : (x < 0 ? complex(log(-x), π) : -Inf)

# Singularity protection functions
function f_inv(x, y_th)
    x_th = (1 ./ y_th)
    return x_th, (y_th ./ x_th) .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x) .* (abs.(x) .>= x_th)
end

function f_inv2(x, y_th)
    x_th = (1 ./ y_th).^0.5
    return x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x.^2) .* (abs.(x) .>= x_th)
end

function f_inv3(x, y_th)
    x_th = 1 ./ y_th.^(1/3)
    return x_th, y_th ./ x_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x.^3) .* (abs.(x) .>= x_th)
end

function f_inv4(x, y_th)
    x_th = 1 ./ y_th.^(1/4)
    return x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x.^4) .* (abs.(x) .>= x_th)
end

function f_inv5(x, y_th)
    x_th = 1 ./ y_th.^(1/5)
    return x_th, y_th ./ x_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ x.^5) .* (abs.(x) .>= x_th)
end

function f_sqrt(x, y_th)
    x_th = 1 ./ y_th.^2
    return x_th, x_th ./ y_th .* x .* (abs.(x) .< x_th) .+ nan_to_num(sqrt.(abs.(x)) .* sign.(x)) .* (abs.(x) .>= x_th)
end

f_power1d5(x, y_th) = abs.(x).^1.5

function f_insqrt(x, y_th)
    x_th = 1 ./ y_th.^2
    return x_th, y_th .* (abs.(x) .< x_th) .+ nan_to_num(1 ./ sqrt.(abs.(x))) .* (abs.(x) .>= x_th)
end

function f_log(x, y_th)
    x_th = exp.(-y_th)
    return x_th, -y_th .* (abs.(x) .< x_th) .+ nan_to_num(log.(abs.(x))) .* (abs.(x) .>= x_th)
end

function f_tan(x, y_th)
    clip = x .% π
    delta = π/2 .- atan.(y_th)
    return clip, delta, -y_th ./ delta .* (clip .- π/2) .* (abs.(clip .- π/2) .< delta) .+ nan_to_num(tan.(clip)) .* (abs.(clip .- π/2) .>= delta)
end

"""
f_arctanh = lambda x, y_th: ((delta := 1-torch.tanh(y_th) + 1e-4), y_th * torch.sign(x) * (torch.abs(x) > 1 - delta) + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta))
f_arcsin = lambda x, y_th: ((), torch.pi/2 * torch.sign(x) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1))
f_arccos = lambda x, y_th: ((), torch.pi/2 * (1-torch.sign(x)) * (torch.abs(x) > 1) + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1))
f_exp = lambda x, y_th: ((x_th := torch.log(y_th)), y_th * (x > x_th) + torch.exp(x) * (x <= x_th))
"""
function f_arctanh(x, y_th)
    delta = 1 .- tanh.(y_th) .+ 1e-4
    return delta, y_th .* sign.(x) .* (abs.(x) .> 1 .- delta) .+ nan_to_num(atanh.(x)) .* (abs.(x) .<= 1 .- delta)
end

function f_arcsin(x, y_th)
    return nothing, π/2 .* sign.(x) .* (abs.(x) .> 1) .+ nan_to_num(asin.(x)) .* (abs.(x) .<= 1)
end

function f_arccos(x, y_th)
    return nothing, π/2 .* (1 .- sign.(x)) .* (abs.(x) .> 1) .+ nan_to_num(acos.(x)) .* (abs.(x) .<= 1)
end

function f_exp(x, y_th)
    x_th = log.(y_th)
    return x_th, y_th .* (x .> x_th) .+ exp.(x) .* (x .<= x_th)
end

end

using Test
using .SymbolicLib: f_inv, f_inv2, f_inv3, f_inv4, f_inv5, f_sqrt, f_power1d5, f_insqrt, f_log, f_tan, f_arctanh, f_arcsin, f_arccos, f_exp

x_test, y_test = [0.1, 0.2], [0.1, 0.2]
@test all(f_inv3(x_test, y_test) .≈ ([2.1544346900318834, 1.7099759466766968], [0.004641588833612781, 0.023392141905702935]))
@test all(f_inv4(x_test, y_test) .≈ ([1.7782794100389228, 1.4953487812212205], [0.1, 0.2]))
@test all(f_inv5(x_test, y_test) .≈ ([1.5848931924611134, 1.379729661461215], [0.006309573444801934, 0.028991186547107823]))
@test all(f_sqrt(x_test, y_test) .≈ ([100.0, 25.0], [100.0, 25.0]))
@test all(f_power1d5(x_test, y_test) ≈ [0.0316227766016838, 0.0894427190999916])
@test all(f_insqrt(x_test, y_test) .≈ ([100.0, 25.0], [0.1, 0.2]))
@test all(f_log(x_test, y_test) .≈ ([0.9048374180359595, 0.8187307530779818], [-0.1, -0.2]))
@test all(f_tan(x_test, y_test) .≈ ([0.1, 0.2], [1.4711276743037345, 1.3734007669450157], [0.09997747663138791, 0.19962073122240753]))
@test all(f_arctanh(x_test, y_test) .≈ ([0.9004320053750442, 0.802724679775096], [0.1, 0.2]))
@test all(f_arcsin(x_test, y_test)[2] ≈ [0.1001674211615598, 0.2013579207903308])
@test all(f_arccos(x_test, y_test)[2] ≈ [1.4706289056333368, 1.369438406004566])
@test all(f_exp(x_test, y_test) .≈ ([-2.3025850929940455, -1.6094379124341003], [0.1, 0.2]))

