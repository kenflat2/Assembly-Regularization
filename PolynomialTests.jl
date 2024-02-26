using Test
include("Polynomial.jl")
include("DifferentialEquations.jl")

# Example temperature schedule (you may customize this based on your problem)
function example_temperature_schedule(iteration)
    return 1.0 / log(1 + iteration)
end

# Example usage
initial_solution = randn(3 )  # Replace with the initial solution for your problem
max_iterations = 100

x1 = randn!(zeros(100))
x2 = randn!(zeros(100))
x3 = randn!(zeros(100))
y = (x1.^2 + x2.^2 + (x3.^2).*2)

model = :((x1.^2 + x2.^2 + (x3.^2).*β))
result_solution, result_energy = simulated_annealing(model)

println("Result Solution: ", result_solution)
println("Result Energy: ", result_energy)
@test result_energy < 0.1

y = x1 .+ x2
model = :(x1 .- β)

β = 5.0
@test compute_predictions(:(β+0.0)) == (ones(100) .* 5.0)
@test compute_predictions(:(x1 .+ 0.0)) == x1

β = 0.0
@test compute_MSE_instantaneous(:(β+0.0)) > 0.5
@test compute_MSE_instantaneous(:(x1 .+ x2)) < 0.5

# Ensure that if a basic building block is the true model, then the 
# first model off the priority queue is that model.
y = x1
init_population()

@test length(P) == 0 && length(Q) == 3

for i in 1:4
    add_to_population(dequeue!(Q))
end

@test length(P) == 3 && length(Q) == 0

add_descendants_to_queue(10000,1000000)

@test length(Q) == 24

x1 = rand(100)
x2 = rand(100)
y = 2 .* x1 .+ 4 .* x2
model = create_descendant(operations[1], (:x1, :x2))
print(model)

# @time P2 = generate_next_generation(P, k)

x1 = u1
x2 = u2
x3 = u3
y = dx1

λ = var(y .- mean(y))
k = 100000

# @time assemble(λ, k)
#=
n1 = AssemblyNode(quote x end, Number, 0, nothing)
n2 = AssemblyNode(:(x+y), Number, 0, n1)

blocks_ = Expr[]
block_types_ = Type[]
get_ancestors(n1, blocks_, block_types_)
@test length(blocks_) == 3
@test length(block_types_) == 3

blocks = Expr[]
block_types = Type[]
get_ancestors(n2, blocks, block_types)
@test length(blocks) == 4
@test length(block_types) == 4

@test blocks[1] == :(x+y)
@test block_types[1] == Number
=#

x1 = randn(100)
x2 = randn(100)
x3 = randn(100)
y = x1

#=
model = :(1.0 .* x1)
@test compute_MI(model) == Inf
model = :(1.0 .* x2)
@test compute_MI(model) < 0.1
model = :(1.0 .* x3)
@test compute_MI(model) < 0.1
y = ones(100)
@test compute_MI(:(5.0+0.0)) == 0.0
=#

#=
# Simplest example
x1 = randn!(zeros(100))
x2 = randn!(zeros(100))
# x3 = randn!(zeros(100))
y = (x1 .* x2)

λ = var(y) / 2

# It finds the correct model here.
@time assemble(λ)


x1 = u1
x2 = u2
y = dx1

λ = var(y .- mean(y)) / 3

@time assemble(λ)
=#

# true_model = :($(building_blocks[1]) .* ($(building_blocks[4]) .- $(building_blocks[3])) .- $(building_blocks[2]))


x1 = u1
x2 = u2
x3 = u3
y = dx3

λ = var(y) / 4

# @time final_model = assemble(λ)

#=
# Test the nonlinear mutual information implementation
x = randn(100) .* 2.0
y = sin.(x.^2)

m1 = :(x .+ 0.0)
m2 = :(x .^ 2)
m3 = :(sin.(x.^2))

MI1 = compute_MI_nonlinear(m1)
MI2 = compute_MI_nonlinear(m2)
MI3 = compute_MI_nonlinear(m3)

@test MI1 < MI2
@test MI2 < MI3

# Test the nonlinear mutual information implementation on the second
# component of the Lorenz63 model

x1 = u1
x2 = u2
x3 = u3
y = dx2

m1 = :(x3 .+ 0.0)
m2 = :(β .- x3)
m3 = :(x1 .* (β .- x3))
m4 = :((x1 .* (β .- x3)) .- x2)

MI1 = compute_MI_nonlinear(m1)
MI2 = compute_MI_nonlinear(m2)
MI3 = compute_MI_nonlinear(m3) 
MI4 = compute_MI_nonlinear(m4)

@test MI1 - MI2 <= 10e-5
@test MI2 - MI3 <= 10e-5
@test MI3 - MI4 <= 10e-5

# compute assembly index of expresssions

e1 = building_blocks[1]
e2 = :($(building_blocks[2]) .* $(building_blocks[1]))
e3 = :(($(building_blocks[2]) .* $(building_blocks[1])) .* ($(building_blocks[2]) .* $(building_blocks[1])))
e4 = :(($(building_blocks[2]) .* $(building_blocks[1])) .* ($(building_blocks[2]) .* $(building_blocks[3])))

@test compute_assembly_index(e1) == 0
@test compute_assembly_index(e2) == 1
@test compute_assembly_index(e3) == 2
@test compute_assembly_index(e4) == 3
=#