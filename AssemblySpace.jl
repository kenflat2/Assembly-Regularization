using Statistics
using Random
include("UnivariateLinear.jl")

struct AssemblyPath
    path::Vector{Expr}
    types::Vector{Type}
end

# Forms the initial population of assembly index 0 models, which are
# simply the building block expression
# Returns - a vector of AssemblyPath objects each initialized with one
#           building block expression
function init_assembly_paths()
    return [AssemblyPath([block], [type]) 
        for (block, type) in zip(building_blocks, building_block_types)
        ]
end;

# Create a new assembly path using new expression, assuming it has
# already been generated from previous elements in the path elsewhere
#   a - assembly path that we wish to augment with a new expression
#   expr - the new expression to augment it with
# Returns: new assembly path with expression added
function create_assembly_path(a::AssemblyPath, expr::Expr)
    new_type = typeof(eval(expr))
    new_path = push!(copy(a.path), expr)
    new_type_list = push!(copy(a.types), new_type)
    return AssemblyPath(new_path, new_type_list)
end;


#=
Can be deleted

# Retrieves all methods from the set of Assembly operations. functions
# may have abstract type arguments while methods implementations
# of that function for each argument signature. We require argument type
# signatures to know how models can be combined under operations.
function get_methods()
    return_m = Vector{Method}()
    for operation in operations
        for method in methods(operation)
            push!(return_m, method)
        end
    end
    return return_m
end;
=#

# Given an assembly path, construct all models what can be assembled
# from it.
#   ap - input assembly path
# returns - Vector of assembly paths of length one greater than the input
function generate_descendants(ap::AssemblyPath)
    assembly_paths = Vector{AssemblyPath}()
    # list all models (both building blocks and those in assembly path)
    # which can be recombined to make new models
    blocks = append!(copy(building_blocks), ap.path)
    block_types = append!(copy(building_block_types), ap.types)

    # for each method, find all combinations of previous models which are
    # valid inputs to it and construct a new assembly path for each
    for operation in operations
        for method in methods(operation)
            for args in arguments_that_match_type_signature(method, blocks, block_types)
                new_assembly_path = create_assembly_path(ap, :($operation($(args...))))
                push!(assembly_paths, new_assembly_path)
            end
        end
    end

    return unique(assembly_paths)
end;

# nice little helper function, credit to 
# https://stackoverflow.com/questions/50899973/indices-of-unique-elements-of-vector-in-julia
function unique_indices(x::Vector)
    return unique(i -> x[i], 1:length(x))
end

# Generate new population given the old population
function generate_next_generation(P::Vector{AssemblyPath})
    P_new = AssemblyPath[]
    for ap in P
        append!(P_new, generate_descendants(ap))
    end

    return P_new
end

# This is working as intended
function arguments_that_match_type_signature(method, blocks, block_types)
    # vector of input types to method 
    method_arg_types = get_method_argument_types(method)
    # for each type, find the models in building blocks and in the
    # assembly path which match that type
    arg_list = Tuple([models_of_type(type, blocks, block_types) 
               for type in method_arg_types])

    # now iterate over the list of each argument in turn
    return Iterators.product(arg_list...)
end;

# Finds all models of a specific type given a vector of models and a 
# list of their types.
#   type - specific type to return
#   blocks - the vector of other models
#   block_types - the return type of each model
# Return - a vector of models
function models_of_type(type, blocks, block_types)
    return blocks[map((x) -> x == type, block_types)]
end;

# Very ugly function that returns the vector of arguments for a 
# method. Don't be surprised if this breaks in a new version of Julia.
function get_method_argument_types(m::Method)
    signature_vector = [x for x in m.sig.parameters]
    return Vector{Type}(signature_vector[2:length(signature_vector)])
end;

# Given a model, some input and output data, compute the MSE
#   a - input assembly path for a model
#   x - inputs to model
#   y - targets
function compute_MSE(a::AssemblyPath, y::Vec)
    m = get_model(a)

    # that is the nicest piece of code in the universe look at that
    # shit mmmmmmmmmmmmmmmmm
    y_hat = eval(m) # the expression is evaluated in the global context
    
    return Statistics.mean((y_hat isa Scalar ? y.vec.-y_hat.val : y.vec-y_hat.vec).^2)
end;

function get_model(a::AssemblyPath)
    return last(a.path)
end

# from a population of models of models with identical assembly index a
# find the best performing one.
#   P - population of assembly paths to evaluate, must have identical assembly index
#   X - n × 1 design matrix, since this is univariate regression
#   y - n × 1 target matrix
#   λ - regularization penalty, must be positive
#   a - current assembly index
function find_best_model(P::Vector{AssemblyPath}, y::Vec, λ::Real, a::Unsigned)
    MSEs = [compute_MSE(ap, y) for ap in P]
    min_index = argmin(MSEs)
    MSEmin = MSEs[min_index]
    best_model = get_model(P[min_index])

    return (best_model, MSEmin + λ*a)
end;

# Assembles the minimal assembly index model for a given dataset
#   X - design matrix
#   y - target matrix
#   λ - regularization penalty
function assemble(y::Vec, λ::Real)
    P = init_assembly_paths()
    a = Unsigned(0)
    (best_model, Lmin) = find_best_model(P, y, λ, a)

    while Lmin >= λ * a
        P = generate_next_generation(P)
        a += 1
        (local_best_model, local_Lmin) = find_best_model(P, y, λ, a)
        if local_Lmin < Lmin
            best_model = local_best_model
            Lmin = local_Lmin
        end
        println(a)
    end

    return best_model
end;