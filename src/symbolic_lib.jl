module SymbolicLib

using Flux, CUDA, KernelAbstractions, LinearAlgebra, SymPy

# Helper functions
nan_to_num(x) = isnan(x) ? zero(x) : x
sign(x) = x < 0 ? -1 : (x > 0 ? 1 : 0)
safe_sqrt(x) = x >= 0 ? sqrt(x) : complex(0, sqrt(-x))
safe_log(x) = x > 0 ? log(x) : (x < 0 ? complex(log(-x), π) : -Inf)

# Singularity protection functions
f_inv(x, y_th) = let x_th = fill(1/y_th, size(x))
    (y_th./x_th .* x) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ x) .* (abs.(x) .>= x_th)
end

f_inv2(x, y_th) = let x_th = fill(1/y_th^(1/2), size(x))
    fill(y_th, size(x)) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ x.^2) .* (abs.(x) .>= x_th)
end

f_inv3(x, y_th) = let x_th = fill(1/y_th^(1/3), size(x))
    (fill(y_th, size(x))./x_th .* x) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ x.^3) .* (abs.(x) .>= x_th)
end

f_inv4(x, y_th) = let x_th = fill(1/y_th^(1/4), size(x))
    fill(y_th, size(x)) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ x.^4) .* (abs.(x) .>= x_th)
end

f_inv5(x, y_th) = let x_th = fill(1/y_th^(1/5), size(x))
    (fill(y_th, size(x))./x_th .* x) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ x.^5) .* (abs.(x) .>= x_th)
end

f_sqrt(x, y_th) = let x_th = fill(1/y_th^2, size(x))
    (x_th./fill(y_th, size(x)) .* x) .* (abs.(x) .< x_th) .+ nan_to_num.(safe_sqrt.(abs.(x)) .* sign.(x)) .* (abs.(x) .>= x_th)
end

f_power1d5(x, y_th) = abs.(x).^1.5 .* sign.(x)

f_invsqrt(x, y_th) = let x_th = fill(1/y_th^2, size(x))
    fill(y_th, size(x)) .* (abs.(x) .< x_th) .+ nan_to_num.(1 ./ safe_sqrt.(abs.(x))) .* (abs.(x) .>= x_th)
end

f_log(x, y_th) = let x_th = fill(exp(-y_th), size(x))
    fill(-y_th, size(x)) .* (abs.(x) .< x_th) .+ nan_to_num.(safe_log.(abs.(x))) .* (abs.(x) .>= x_th)
end

f_tan(x, y_th) = let clip = x .% π, delta = fill(π/2 - atan(y_th), size(x))
    (-fill(y_th, size(x))./delta .* (clip .- π/2)) .* (abs.(clip .- π/2) .< delta) .+ nan_to_num.(tan.(clip)) .* (abs.(clip .- π/2) .>= delta)
end

f_arctanh(x, y_th) = let delta = fill(1 - tanh(y_th) + 1e-4, size(x))
    fill(y_th, size(x)) .* sign.(x) .* (abs.(x) .> 1 .- delta) .+ nan_to_num.(atanh.(clamp.(x, -1+1e-7, 1-1e-7))) .* (abs.(x) .<= 1 .- delta)
end

f_arcsin(x, y_th) = (π/2) .* sign.(x) .* (abs.(x) .> 1) .+ nan_to_num.(asin.(clamp.(x, -1, 1))) .* (abs.(x) .<= 1)

f_arccos(x, y_th) = (π/2) .* (1 .- sign.(x)) .* (abs.(x) .> 1) .+ nan_to_num.(acos.(clamp.(x, -1, 1))) .* (abs.(x) .<= 1)

f_exp(x, y_th) = let x_th = fill(log(y_th), size(x))
    fill(y_th, size(x)) .* (x .> x_th) .+ exp.(x) .* (x .<= x_th)
end

# Define the SYMBOLIC_LIB dictionary
SYMBOLIC_LIB = Dict(
    "x" => (x -> x, x -> x, 1, (x, y_th) -> x),
    "x^2" => (x -> x.^2, x -> x^2, 2, (x, y_th) -> x.^2),
    "x^3" => (x -> x.^3, x -> x^3, 3, (x, y_th) -> x.^3),
    "x^4" => (x -> x.^4, x -> x^4, 3, (x, y_th) -> x.^4),
    "x^5" => (x -> x.^5, x -> x^5, 3, (x, y_th) -> x.^5),
    "1/x" => (x -> 1 ./ x, x -> 1/x, 2, f_inv),
    "1/x^2" => (x -> 1 ./ x.^2, x -> 1/x^2, 2, f_inv2),
    "1/x^3" => (x -> 1 ./ x.^3, x -> 1/x^3, 3, f_inv3),
    "1/x^4" => (x -> 1 ./ x.^4, x -> 1/x^4, 4, f_inv4),
    "1/x^5" => (x -> 1 ./ x.^5, x -> 1/x^5, 5, f_inv5),
    "sqrt" => (x -> safe_sqrt.(x), x -> sqrt(x), 2, f_sqrt),
    "x^0.5" => (x -> safe_sqrt.(x), x -> sqrt(x), 2, f_sqrt),
    "x^1.5" => (x -> safe_sqrt.(x).^3, x -> sqrt(x)^3, 4, f_power1d5),
    "1/sqrt(x)" => (x -> 1 ./ safe_sqrt.(x), x -> 1/sqrt(x), 2, f_invsqrt),
    "1/x^0.5" => (x -> 1 ./ safe_sqrt.(x), x -> 1/sqrt(x), 2, f_invsqrt),
    "exp" => (x -> exp.(x), x -> exp(x), 2, f_exp),
    "log" => (x -> safe_log.(x), x -> log(x), 2, f_log),
    "abs" => (x -> abs.(x), x -> abs(x), 3, (x, y_th) -> abs.(x)),
    "sin" => (x -> sin.(x), x -> sin(x), 2, (x, y_th) -> sin.(x)),
    "cos" => (x -> cos.(x), x -> cos(x), 2, (x, y_th) -> cos.(x)),
    "tan" => (x -> tan.(x), x -> tan(x), 3, f_tan),
    "tanh" => (x -> tanh.(x), x -> tanh(x), 3, (x, y_th) -> tanh.(x)),
    "sgn" => (x -> sign.(x), x -> sign.(x), 3, (x, y_th) -> sign.(x)),
    "arcsin" => (x -> asin.(clamp.(x, -1, 1)), x -> asin(x), 4, f_arcsin),
    "arccos" => (x -> acos.(clamp.(x, -1, 1)), x -> acos(x), 4, f_arccos),
    "arctan" => (x -> atan.(x), x -> atan(x), 4, (x, y_th) -> atan.(x)),
    "arctanh" => (x -> atanh.(clamp.(x, -1+1e-7, 1-1e-7)), x -> atanh(x), 4, f_arctanh),
    "0" => (x -> x .* 0, x -> x * 0, 0, (x, y_th) -> x .* 0),
    "gaussian" => (x -> exp.(-x.^2), x -> exp(-x^2), 3, (x, y_th) -> exp.(-x.^2)),
)

# Test the symbolic library
x = randn(10)
y_th = 1e-3
for (key, (f, f_str, order, f_inv)) in SYMBOLIC_LIB
    println("Testing $key")
    x_sym = f(x)
    x_inv = f_inv(x_sym, y_th)
    println("x_sym: $x_sym")
    println("x_inv: $x_inv")
    println()
end

end

# Test
using .SymbolicLib

# Test the symbolic library
x = randn(10)
y_th = 1e-3
for (key, (f, f_str, order, f_inv)) in SymbolicLib.SYMBOLIC_LIB
    println("Testing $key")
    x_sym = f(x)
    x_inv = f_inv(x_sym, y_th)
    println("x_sym: $x_sym")
    println("x_inv: $x_inv")
end