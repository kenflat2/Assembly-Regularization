using Statistics
using Random
include("AssemblySpace.jl")

global x1::Vector{Float64}
global x2::Vector{Float64}
global x3::Vector{Float64}
global y::Vector{Float64}
global σ = 0::Number
global ρ = 0::Number
global β = 0::Number

# building_blocks = Vector{Expr}([quote x1 end, quote x2 end, quote x3 end, quote σ end, quote ρ end, quote β end]);
building_blocks = Vector{Expr}([quote x1 end, quote x2 end, quote x3 end, quote σ end, quote ρ end, quote β end]);
building_block_types = Vector{Type}([Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64]);
operations = Vector([.+, .-, .*]);
operation_input_types = [[(Vector{Float64}, Vector{Float64}), (Float64, Vector{Float64})],
                         [(Vector{Float64}, Vector{Float64}), (Float64, Vector{Float64})],
                         [(Vector{Float64}, Vector{Float64}), (Float64, Vector{Float64})]]

# Computes the MSE after optimizing scalar-valued parameters
function compute_MSE(m::Expr)
    (min_solution, min_energy) = simulated_annealing(m)

    return min_energy
end

# Given a model, some input and output data, compute the MSE without
# optimizing the scalar parameters
#   m - input model
function compute_MSE_instantaneous(m::Expr)
    # that is the nicest piece of code in the universe look at that
    # shit mmmmmmmmmmmmmmmmm
    y_hat = eval(m) # the expression is evaluated in the global context
    
    return Statistics.mean((y_hat isa Number ? y_hat .- y : y - y_hat).^2)
end;

function simulated_annealing(m::Expr,
                             initial_solution = (0.0,0.0,0.0), 
                             temperature_schedule = (iteration -> 1.0 / log(1 + iteration)),
                             max_iterations = 1000)

    global σ
    global ρ
    global β

    temperature_schedule = iteration -> 1.0 / log(1 + iteration)

    current_solution = initial_solution
    (σ, ρ, β) = current_solution
    current_energy = compute_MSE_instantaneous(m)

    min_solution = current_solution
    min_energy = current_energy

    for iteration in 1:max_iterations
        temperature = temperature_schedule(iteration)

        # Generate a neighboring solution
        candidate_solution = generate_neighbor(current_solution)
        (σ, ρ, β) = candidate_solution

        # Calculate the energy of the candidate solution
        candidate_energy = compute_MSE_instantaneous(m)

        # Accept the candidate solution if it has lower energy or with a certain probability
        if candidate_energy < current_energy || rand() < exp((current_energy - candidate_energy) / temperature)
            current_solution = candidate_solution
            current_energy = candidate_energy
            if current_energy < min_energy
                min_energy = current_energy
                min_solution = current_solution
            end
        end
    end

    return min_solution, min_energy
end

# Function to generate a neighboring solution (you may customize this based on your problem)
function generate_neighbor(current_solution)
    return current_solution .+ randn(3)
end