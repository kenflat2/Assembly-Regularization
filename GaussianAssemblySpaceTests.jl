using Test
using Plots

include("GaussianAssemblySpace.jl")
include("DifferentialEquations.jl")
include("Polynomial.jl")

x1 = u1
x2 = u2
x3 = u3
o = ones(length(x1))
y = dx2

expr_inputs = linear_combination_of_inputs(building_blocks)
@test expr_inputs == :(((w[1] .* x1 .+ w[2] .* x2) .+ w[3] .* x3) .+ w[4] .* o)

expr_inputs = linear_combination_of_inputs(building_blocks, 5)
@test expr_inputs == :(((w[6] .* x1 .+ w[7] .* x2) .+ w[8] .* x3) .+ w[9] .* o)

inputs = Vector{Term}([:x1])
expr_inputs = linear_combination_of_inputs(inputs)
@test expr_inputs == :(w[1] .* x1)

inputs = Vector{Term}()
expr_inputs = linear_combination_of_inputs(inputs)
@test expr_inputs == :()

# Define some example functions
f1(x) = sin.(x)
f2(x) = x.^2
f3(x) = exp.(x)

n_datapoints = 200
x1 = randn(n_datapoints)
x2 = randn(n_datapoints)
x3 = randn(n_datapoints)
o = ones(n_datapoints)
y = f1.(x1)

# Create an array of functions
functions = [f1, f2]

# Generate the linear combination expression
result_expression, w_size = linear_combination_of_functions(building_blocks, functions)

# test that passing in a vector works
w = rand(2)
expr = :(w[1] .* o + w[2] .* o)
@test w[1] + w[2] == (eval(expr))[1]
@test w_size == 10

w = zeros(10)
w[1] = 1.0
w[2] = 1.0
@test eval(result_expression)[1] == f1(x1[1])

functions = [f1, f2, f3]
result_expression, w_size = linear_combination_of_functions(building_blocks, functions)

@test w_size == 15

# see if we get a Gaussian distribution of errors
ensemble_size = 1000
ensemble_variance = 2

ensemble = randn((w_size, ensemble_size)) .* ensemble_variance
errors = zeros(ensemble_size)

for i in 1:ensemble_size
    global w = ensemble[:,i]

    errors[i] = log(mean((eval(result_expression) .- y) .^ 2))
end

MI = [-1 * log(1 - cor(ensemble[i,:], errors)^2) / 2 for i in 1:w_size]

# Optimization test 1: given constant input and constant output, will it
# learn a weight of 5?

n = 10
x1 = randn(n)
x2 = randn(n)
x3 = randn(n)
model = :(w[1] .* x1)
y = 5 .* x1

ensemble_size = 100
ensemble = randn((1, ensemble_size))

# generate_forecasts test

forecasts = generate_forecasts(ensemble, model, ensemble_size)
@test length(forecasts) == ensemble_size
@test forecasts[1] == ensemble[1, 1] * x1[1]

observation = y[1]
innovation = observation .- forecasts
@test innovation[1] == observation - forecasts[1]

forecasts_st = forecasts .- mean(forecasts)
ensemble_st = ensemble .- mean(ensemble, dims = 2)
@test forecasts_st[1] == forecasts[1] - mean(forecasts)

kalman_gain = (ensemble_st * forecasts_st) .* inv(forecasts_st' * forecasts_st + 1)
@test isapprox(kalman_gain[1], (ensemble_st[1,:]' * forecasts_st) * inv(forecasts_st' * forecasts_st+1))

ensemble2 = ensemble .+ (kalman_gain * innovation')

# update_step!(ensemble, observation, model, ensemble_size)
enkf!(ensemble, model, ensemble_size)
@test isapprox(5, mean(ensemble), atol=1)

# Optimization test 2: 

n = 200
x1 = randn(n)
x2 = randn(n)
x3 = randn(n)
o = ones(n)
model = result_expression
y = f1.(2 .* x1 .- o)

ensemble_size = 100
ensemble = randn((15, ensemble_size))

enkf!(ensemble, model, ensemble_size)