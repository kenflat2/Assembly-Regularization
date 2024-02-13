using Test
using Random
include("Polynomial.jl")
include("DifferentialEquations.jl")

# Example temperature schedule (you may customize this based on your problem)
function example_temperature_schedule(iteration)
    return 1.0 / log(1 + iteration)
end

# Example usage
initial_solution = randn(3 )  # Replace with the initial solution for your problem
max_iterations = 10000

x1 = randn!(zeros(100))
x2 = randn!(zeros(100))
x3 = randn!(zeros(100))
y = (x1.^2 + x2.^2 + (x3.^2).*2)

model = :((x1.^2 + x2.^2 + (x3.^2).*σ))
result_solution, result_energy = simulated_annealing(model)

println("Result Solution: ", result_solution)
println("Result Energy: ", result_energy)
@test result_energy < 0.1

k = 100
y = x1 .+ x2
model = :(x1 .- σ)

@time P = init_population(10)
@test get_model(peek(P.pq)[1]) == building_blocks[3]

@time P2 = generate_next_generation(P, k)

x1 = u1
x2 = u2
x3 = u3
y = dx1

λ = 10000
k = 100000

# @time assemble(λ, k)