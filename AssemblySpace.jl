using Statistics
using Random
using Plots
include("DataStructures.jl")

# const Population = FixedSizePriorityQueue{AssemblyPath, Real}
const Population = Set{Term}
const BirthQueue = PriorityQueue{Term, Real}

# Forms the initial population of assembly index 0 models, which are
# simply the building block expression
# Returns - a vector of AssemblyPath objects each initialized with one
#           building block expression
function init_population()
    global P
    global Q
    global P_types

    P = Set{Term}()
    Q = BirthQueue(Base.Order.Reverse)
    P_types = Dict{Term, Type}()

    for (block, type) in zip(building_blocks, building_block_types)
        mi = 1000
        enqueue!(Q, block, mi)
    end
end;

# The ultimate function
function assemble(λ::Real)
    @time init_population()

    minimal_model = building_blocks[1]
    minimal_loss = Inf

    while !isempty(Q)
        # pop a model off the queue and assess its MSE
        model = dequeue!(Q)

        print_poly(model)

        loss = compute_loss(model, λ)
        # print(model, "j\n")
        if loss < minimal_loss
            minimal_loss = loss
            minimal_model = model
            print("Best Model: ")
            print_poly(model)
            println("")
        end

        if minimal_loss < 2845
            return minimal_model
        end

        # if this new model has a probitively large assembly, then don't add it to the population
        if minimal_loss < λ * compute_assembly_index(model)
            continue
        end

        # then push to the population set so it is never recomputed
        add_to_population(model)

        # then add descendants to the queue, sorting them by their mutual information
        add_descendants_to_queue(λ, minimal_loss)
    end

    return minimal_model
end

# add a model to the population of models so it is never recomputed
function add_to_population(model::Term)
    push!(P, model)
    P_types[model] = typeof(eval(model))
end

function add_descendants_to_queue(λ::Real, minimal_loss::Real)
    descendants = generate_descendants()

    for child_model in descendants
        # ensure child hasn't been checked
        reg_penalty = λ * compute_assembly_index(child_model)
        if !(child_model in P) && !(child_model in keys(Q)) && (reg_penalty < minimal_loss)
            child_MI = compute_MI(child_model)
            enqueue!(Q, child_model, child_MI)
        end
    end
end

# Computes the assembly index of a generic Julia expression, assuming the set of building blocks.
# The assembly index is equal to the number of unique models in the assembly path which are not 
# building blocks. For instance
#  a -> a + b -> (a + b)^2
# The arguments to the ultimate expression are both (a + b), 
function compute_assembly_index(model::Term)
    set_of_models = Set{Term}()
    set_of_models_in_path(model, set_of_models)

    # return the number of non-building block models in the assembly path
    return length(set_of_models)
end

# Helper function to compute the assembly index
function set_of_models_in_path(model::Term, set_of_models::Set{Term})
    # if the expression is a building block, skip
    if !(model in building_blocks)
        # the expression is an intermediate model in the assembly path, so add it to the set.
        push!(set_of_models, model)

        # repeat for each input argument expression.
        for arg in model.args[2:length(model.args)]
            set_of_models_in_path(arg, set_of_models)
        end
    end
end

#=
function get_ancestors(model::Term, blocks::Vector{Term}, block_types::Vector{Type})
    if node.parent === nothing
        append!(blocks, building_blocks)
        append!(block_types, building_block_types)
    else
        push!(blocks, node.model)
        push!(block_types, node.type)
        get_ancestors(node.parent, blocks, block_types)
    end
end
=#

function generate_descendants()
    # the population is all models which can be recombined
    descendants = Term[]
    
    # for each method, find all combinations of previous models which are
    # valid inputs to it and construct a new assembly path for each
    for (operation, op_signatures) in zip(operations, operation_input_types)
        for op_signature in op_signatures
            for args in arguments_that_match_type_signature(op_signature)
                descendant = create_descendant(operation, args)
                push!(descendants, descendant)
            end
        end
    end

    return descendants
end;

# create a descendant depending on the operation and arguments passed in
function create_descendant(operation, args)
    if operation == Base.Broadcast.BroadcastFunction(+)
        in1 = eval(args[1])
        in2 = eval(args[2])

        # do linear regression manually because its only two variables who needs libraries.
        coef =  1/ (dot(in1, in1) + dot(in2, in2))
        a = coef * dot(in1, y)
        b = coef * dot(in2, y)

        return :($a .* arg[1] .+ $b .* arg[2])
    end
    return :($operation($(args...)))
end

# This is working as intended
function arguments_that_match_type_signature(op_signature)
    # for each type, find the models in building blocks and in the
    # assembly path which match that type
    arg_list = Tuple([models_of_type(type) for type in op_signature])

    # now iterate over the list of each argument in turn
    return Iterators.product(arg_list...)
end;

function compute_loss(model::Term, λ::Real)
    assembly_index = compute_assembly_index(model)
    return compute_MSE(model) + λ * assembly_index
end;

# Finds all models of a specific type given a vector of models and a 
# list of their types.
#   type - specific type to return
#   blocks - the vector of other models
#   block_types - the return type of each model
# Return - a vector of models
function models_of_type(type)
    return filter((m) -> P_types[m] == type, P)
end;

# Print expression
function print_expression(expr::Term, func_mappings::Dict=Dict(), var_mappings::Dict=Dict())
    if typeof(expr) == Symbol
        
        var_name = get(var_mappings, expr, expr)

        print(var_name)

    elseif expr.head == :call
        # Check if the function has a custom mapping
        func_name = get(func_mappings, expr.args[1], expr.args[1])

        # Print the function name
        print("(")

        print_expression(expr.args[2], func_mappings, var_mappings)
        print("$func_name")
        print_expression(expr.args[3], func_mappings, var_mappings)

        print(")")
    end
end

#=
# Create a new assembly path using new expression, assuming it has
# already been generated from previous elements in the path elsewhere
#   a - assembly path that we wish to augment with a new expression
#   expr - the new expression to augment it with
# Returns: new assembly path with expression added
function create_assembly_path(a::AssemblyPath, expr::Term)
    new_type = typeof(eval(expr))
    new_path = push!(copy(a.path), expr)
    new_type_list = push!(copy(a.types), new_type)
    return AssemblyPath(new_path, new_type_list)
end;
=#

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

#=
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
=#

#=
# Very ugly function that returns the vector of arguments for a 
# method. Don't be surprised if this breaks in a new version of Julia.
function get_method_argument_types(m::Method)
    signature_vector = [x for x in m.sig.parameters]
    return Vector{Type}(signature_vector[2:length(signature_vector)])
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

=#