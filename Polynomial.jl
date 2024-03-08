using Statistics
using Random
using LinearAlgebra
include("AssemblySpace.jl")

global x1::Vector{Float64}
global x2::Vector{Float64}
global x3::Vector{Float64}
global y::Vector{Float64}
# global β = 0::Number

# building_blocks = Vector{Term}([quote x1 end, quote x2 end, quote x3 end, quote σ end, quote ρ end, quote β end]);
# building_blocks = Vector{Term}([quote x1 end, quote x2 end, quote x3 end, quote σ end, quote ρ end, quote β end]);
building_blocks = Vector{Term}([:x1, :x2, :x3, :o]);
# building_block_types = Vector{Type}([Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64]);
building_block_types = Vector{Type}([Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}]);
# building_blocks = Vector{Term}([:(x1 .+ 0.0), :(x2 .+ 0.0), :(β + 0.0)]);
# building_block_types = Vector{Type}([Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64, Float64, Float64]);
# building_block_types = Vector{Type}([Vector{Float64}, Vector{Float64}, Float64]);
operations = Vector([.+, .*]);
operation_input_types = [[(Vector{Float64}, Vector{Float64}), (Float64, Vector{Float64})],
                         [(Vector{Float64}, Vector{Float64}), (Float64, Vector{Float64})]]

# Computes the MSE after optimizing scalar-valued parameters
function compute_MSE(m::Term)
    # (min_solution, min_energy) = simulated_annealing(m)

    return compute_MSE_instantaneous(m) #min_energy
end

function compute_predictions(m::Term)
    y_hat = eval(m)

    if typeof(y_hat) == Float64
        return ones(length(x1)) .* y_hat
    end

    return y_hat
end

# Given a model, some input and output data, compute the MSE without
# optimizing the scalar parameters
#   m - input model
function compute_MSE_instantaneous(m::Term)
    # that is the nicest piece of code in the universe look at that
    # shit mmmmmmmmmmmmmmmmm
    y_hat = compute_predictions(m) # the Termession is evaluated in the global context
    
    return mean((y_hat .- y).^2)
end;

function simulated_annealing(model::Term,
                             max_iterations = 100,
                             initial_solution = 0.0, 
                             temperature_schedule = (iteration -> 1.0 / log(1 + iteration)))
    global β

    temperature_schedule = iteration -> 1.0 / log(1 + iteration)

    current_solution = initial_solution
    β = current_solution
    current_energy = compute_MSE_instantaneous(model)

    min_solution = current_solution
    min_energy = current_energy

    for iteration in 1:max_iterations
        temperature = temperature_schedule(iteration)

        # Generate a neighboring solution
        candidate_solution = generate_neighbor(current_solution)
        β = candidate_solution

        # Calculate the energy of the candidate solution
        candidate_energy = compute_MSE_instantaneous(model)

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


# Null Hypothesis
function compute_MI(model::Term)
    # Find the correct parameters
    return 0
end

#=
# Alt Hypothesis 1
# Mutual information computation assuming the random variables are
# joint Gaussian and the MI can be computed using the correlation.
function compute_MI(model::Term)
    # Find the correct parameters
    # simulated_annealing(model)

    y_hat = compute_predictions(model)
    correlation = cor(y_hat, y)
    correlation = isnan(correlation) ? 0 : correlation
    return -1 / 2 * log(1 - correlation^2)
end
=#
#=
# Alt Hypothesis 2
# Mutual information computation under a flaky assumption of analogue forecasting skill
function compute_MI(model::Term)
    # Find the correct parameters
    simulated_annealing(model)

    y_hat = compute_predictions(model)

    # to maximize computational simplicity, we sort both y and y_hat according
    # to y_hat and take the nearest neighbor as the next element in y_hat.
    # The residuals are then the squared differences between adjacent elements
    # in y.
    ordering = sortperm(y_hat)

    y_ord = y[ordering]
    l = length(y_ord)
    return -mean((y_ord[1:l-1] .- y_ord[2:l]).^2)
end
=#
# Function to generate a neighboring solution (you may customize this based on your problem)
function generate_neighbor(current_solution)
    return current_solution + randn(1)[1]
end

function print_poly(expr::Term)
    func_dict = Dict(:.* => " * ", :.- => " - ", :.+ => " + ", Base.Broadcast.BroadcastFunction(-) => " - ", Base.Broadcast.BroadcastFunction(*) => " * ")
    var_dict = Dict(:x1 => "x", :x2 => "y", :x3 => "z")

    print_expression(expr, func_dict, var_dict)

    println("")
end