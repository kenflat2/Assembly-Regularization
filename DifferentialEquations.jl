using DifferentialEquations
using Plots

function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

function lorenz63(x1::Vector{Float64}, x2::Vector{Float64}, x3::Vector{Float64})
    return (σ .* (x2 .- x1), x1 .* (ρ .- x3) .- x2, x1 .* x2 - (β .* x3))
end

# Initial conditions
u0 = [1.0, 0.0, 0.0]

# Parameters
σ = 10.0
ρ = 28.0
β = 8/3
params = [σ, ρ, β]

# Time span
tspan = (0.0, 30.0)

# Define the ODE problem
prob = ODEProblem(lorenz63!, u0, tspan, params)

# Solve the ODE problem
sol = solve(prob)

u1 = hcat(sol.u...)[1,:]
u2 = hcat(sol.u...)[2,:]
u3 = hcat(sol.u...)[3,:]

# Plot the solution
# plot(sol, vars=(1, 2, 3), xlabel="x", ylabel="y", zlabel="z", legend=false)

(dx1, dx2, dx3) = lorenz63(u1, u2, u3)