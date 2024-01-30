include("UnivariateLinear.jl")

struct AssemblyPath
    path::Vector{Union{Expr, Symbol}}
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
    new_path = push!(a.path, expr)
    new_type_list = push!(a.types, new_type)
    return AssemblyPath(new_path, new_type_list)
end;

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

# Given an assembly path, construct all models what can be assembled
# from it.
#   ap - input assembly path
# returns - Vector of assembly paths of length one greater than the input
function generate_descendents(ap::AssemblyPath)
    assembly_paths = Vector{AssemblyPath}()
    # list all models (both building blocks and those in assembly path)
    # which can be recombined to make new models
    blocks = push!(building_blocks, ap.path)
    block_types = push!(building_block_types, ap.types)

    # for each method, find all combinations of previous models which are
    # valid inputs to it and construct a new assembly path for each
    for method in get_methods()
        for args in arguments_that_match_type_signature(method, blocks, block_types)
            new_assembly_path = create_assembly_path(ap, :(method(args...)))
            push!(assembly_paths, new_assembly_path)
        end
    end

    return assembly_paths
end

function arguments_that_match_type_signature(method, blocks, block_types)
    # vector of input types to method 
    method_arg_types = get_method_argument_types(method)

    # for each type, find the models in building blocks and in the
    # assembly path which match that type
    arg_list = Tuple([models_of_type(type, blocks, block_types ) 
               for type in method_arg_types])

    # now iterate over the list of each argument in turn
    return Iterators.product(arg_list...)
end;

# Finds all models of a specific type given a vector of models and a 
# list of their types.
#   type - specific type to return
#   blocks - the vector of other models
#   block_types - the return type of each model
function models_of_type(type, blocks, block_types)
    return blocks[map((x) -> x == type, block_types)]
end;

# Very ugly function that returns the vector of arguments for a 
# method. Don't be surprised if this breaks in a new version of Julia.
function get_method_argument_types(m::Method)
    signature_vector = [x for x in get_methods()[1].sig.parameters]
    return Vector{Type}(signature_vector[2:length(signature_vector)])
end;