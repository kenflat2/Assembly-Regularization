using Statistics
using Random

include("AssemblySpace.jl")

# generate an expression that is a linear combination of the current population of 
# assembled models.
# Arguments
#   terms - list of all expression available for assembly
#   weight_index - index to begin counting new weights from
function linear_combination_of_inputs(terms::Vector{Term}, weight_index = 0)
    n = length(terms)
    if n == 0
        return :()
    end
    
    result_expr = :(w[$(weight_index+1)] .* $(terms[1]))
    
    for i in 2:n
        i_updated = weight_index + i
        result_expr = :($result_expr .+ w[$i_updated] .* $(terms[i]))
    end
    
    return result_expr
end

# This creates the expression which is a linear combination of all functions applied to 
# linear combinations of all inputs. This represents the complete space of possible next
# models that could be assembled in the next assembly step. 
#   terms - list of possible arguments to the function
#   functions - list of function
function linear_combination_of_functions(terms::Vector{Term}, functions::Vector{Function}, weight_index = 0)
    n = length(functions)
    n_terms = length(terms)
    
    input_expression = linear_combination_of_inputs(terms, weight_index + 1)
    
    result_expr = :(w[1] .* $(functions[1])( $input_expression ))
    
    for i in 2:n
        weight_index2 = (i - 1) * n_terms + i
        input_expression = linear_combination_of_inputs(terms, weight_index2)
        result_expr = :($result_expr + w[$(weight_index2)] .* $(functions[i])( $input_expression ))
    end
    
    return (result_expr, n * n_terms + n)
end

# Ensemble Kalman Filter Update Step
#   A - (p, E) ensemble of parameter values
#   model - the model that is evaluated with the parameter ensemble
#   observation - the particular observation we are looking for
function enkf!(ensemble, model, ensemble_size)

    global x1, x2, x3

    # copy the input values
    x1_copy = copy(x1)
    x2_copy = copy(x2)
    x3_copy = copy(x3)
    
    # Loop over dataset
    for i in eachindex(y)

        x1 = x1_copy[i, :]
        x2 = x2_copy[i, :]
        x3 = x3_copy[i, :]

        observation = y[i]

        # for each individual observation, update the ensemble
        update_step!(ensemble, observation, model, ensemble_size)
    end

    x1 = x1_copy
    x2 = x2_copy
    x3 = x3_copy
end

# Performs a single ENKF update of the ensemble
function update_step!(ensemble, observation, model, ensemble_size)
    forecasts = generate_forecasts(ensemble, model, ensemble_size)

    innovation = observation .- forecasts
    forecasts_st = forecasts .- mean(forecasts)
    ensemble_st = ensemble .- mean(ensemble, dims = 2)

    kalman_gain = (ensemble_st * forecasts_st) .* inv(forecasts_st' * forecasts_st + 1)

    ensemble_update = (kalman_gain * innovation')
    for i in 1:length(ensemble[:,1])
        for j in 1:ensemble_size
            ensemble[i,j] += ensemble_update[i,j]
        end
    end
end

# Computes the forecasts for a given focal model on a single n_datapoints
# Note: x1, x2, x3 MUST be univariate
function generate_forecasts(ensemble, model, ensemble_size)
    forecasts = zeros(ensemble_size)

    # generate forecasts for each ensemble member
    for j in 1:ensemble_size
        global w = ensemble[:,j]
        # this is inefficient, we evaluate the model on the vector of 
        # all inputs then only take the ith.

        forecasts[j] = eval(model)[1]
    end

    return forecasts
end