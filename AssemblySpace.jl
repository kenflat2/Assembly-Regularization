using Statistics
using Random
include("DataStructures.jl")

struct AssemblyPath
    path::Vector{Expr}
    types::Vector{Type}
end

const Population = FixedSizePriorityQueue{AssemblyPath, Real}

# Forms the initial population of assembly index 0 models, which are
# simply the building block expression
# Returns - a vector of AssemblyPath objects each initialized with one
#           building block expression
function init_population(k::Int)
    P = Population(k)

    for (block, type) in zip(building_blocks, building_block_types)
        new_ap = AssemblyPath([block], [type])
        add_to_population(new_ap, P)
    end

    return P
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
function generate_descendants(ap::AssemblyPath, P::Population)
    # list all models (both building blocks and those in assembly path)
    # which can be recombined to make new models
    blocks = append!(copy(building_blocks), ap.path)
    block_types = append!(copy(building_block_types), ap.types)

    # for each method, find all combinations of previous models which are
    # valid inputs to it and construct a new assembly path for each
    for (operation, op_signatures) in zip(operations, operation_input_types)
        for op_signature in op_signatures
            for args in arguments_that_match_type_signature(op_signature, blocks, block_types)
                new_assembly_path = create_assembly_path(ap, :($operation($(args...))))
                add_to_population(new_assembly_path, P)
            end
        end
    end
end;

function add_to_population(ap::AssemblyPath, P::Population)
    m = get_model(ap)
    MSE = compute_MSE(m)
    enqueue!(P, ap, MSE)
end

# nice little helper function, credit to 
# https://stackoverflow.com/questions/50899973/indices-of-unique-elements-of-vector-in-julia
function unique_indices(x::Vector)
    return unique(i -> x[i], 1:length(x))
end

# Generate new population given the old population
function generate_next_generation(P::Population, k::Int)
    P_new = Population(k)

    for ap in keys(P.pq)
        generate_descendants(ap, P_new)
    end

    return P_new
end

# This is working as intended
function arguments_that_match_type_signature(op_signature, blocks, block_types)
    # for each type, find the models in building blocks and in the
    # assembly path which match that type
    arg_list = Tuple([models_of_type(type, blocks, block_types) 
               for type in op_signature])

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

#=
# Very ugly function that returns the vector of arguments for a 
# method. Don't be surprised if this breaks in a new version of Julia.
function get_method_argument_types(m::Method)
    signature_vector = [x for x in m.sig.parameters]
    return Vector{Type}(signature_vector[2:length(signature_vector)])
end;
=#

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
function find_best_model(P::Population, λ::Real, a::Unsigned)
    best_assembly_path = argmin(P.pq)
    MSEmin = P.pq[best_assembly_path]
    best_model = get_model(best_assembly_path)
    return (best_model, MSEmin + λ*a)
end;

# Assembles the minimal assembly index model for a given dataset. You 
# must set the design matrix x and target matrix y in a global context
# outside of the call. This enables the models to be Julia expresssions.
#   λ - regularization penalty
#   k - carrying capacity
function assemble(λ::Real, k::Int)
    @time P = init_population(k)
    a = Unsigned(0)
    @time (best_model, Lmin) = find_best_model(P,  λ, a)

    while Lmin >= λ * a
        @time P = generate_next_generation(P, k)
        a += 1
        @time (local_best_model, local_Lmin) = find_best_model(P, λ, a)
        if local_Lmin < Lmin
            best_model = local_best_model
            Lmin = local_Lmin
            print(local_best_model)
        end
        println(a)
    end

    return best_model
end;