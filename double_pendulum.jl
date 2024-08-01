using DifferentialEquations, Plots

# Lagrangian of a double pendulum
function double_pendulum!(du, u, p, t)
    g, m1, m2, L1, L2 = p
    θ1, ω1, θ2, ω2 = u
    
    Δθ = θ2 - θ1
    denom1 = L1 * (m1 + m2 * sin(Δθ)^2)
    denom2 = L2 * (m1 + m2 * sin(Δθ)^2)
    
    du[1] = ω1
    du[2] = (m2 * L1 * ω1^2 * sin(Δθ) * cos(Δθ) + m2 * g * sin(θ2) * cos(Δθ) + m2 * L2 * ω2^2 * sin(Δθ) - (m1 + m2) * g * sin(θ1)) / denom1
    du[3] = ω2
    du[4] = (-m2 * L2 * ω2^2 * sin(Δθ) * cos(Δθ) + (m1 + m2) * g * sin(θ1) * cos(Δθ) - (m1 + m2) * L1 * ω1^2 * sin(Δθ) - (m1 + m2) * g * sin(θ2)) / denom2
end

# Parameters and initial conditions
g = 9.81
m1, m2 = 1.0, 1.0
L1, L2 = 1.0, 1.0
θ1, ω1, θ2, ω2 = π/2, 0.0, π/2, 0.0
p = (g, m1, m2, L1, L2)
u0 = [θ1, ω1, θ2, ω2]
tspan = (0.0, 20.0)

prob = ODEProblem(double_pendulum!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

function pendulum_positions(sol, L1, L2)
    x1 = L1 * sin.(sol[1, :])
    y1 = -L1 * cos.(sol[1, :])
    x2 = x1 + L2 * sin.(sol[3, :])
    y2 = y1 - L2 * cos.(sol[3, :])
    return x1, y1, x2, y2
end

x1, y1, x2, y2 = pendulum_positions(sol, L1, L2)

plot_size = (800, 800)
lim = 2.2
trail_length = 100

anim = @animate for i in 1:length(sol.t)
    plot(size=plot_size, aspect_ratio=:equal, xlim=(-lim, lim), ylim=(-lim, lim),
         legend=false, grid=false, axis=false, title="Double Pendulum Simulation")
    
    # Trail
    trail_start = max(1, i - trail_length)
    plot!(x2[trail_start:i], y2[trail_start:i], color=:lightblue, alpha=0.5, lw=2)
    
    # Pendulum rods
    plot!([0, x1[i], x2[i]], [0, y1[i], y2[i]], color=:black, lw=2)
    
    # Pendulum bobs
    scatter!([0, x1[i], x2[i]], [0, y1[i], y2[i]], 
             color=[:gray, :red, :blue], markersize=[6, 10, 10], 
             markerstrokewidth=0)
end

gif(anim, "figures/double_pendulum.gif", fps=30)

