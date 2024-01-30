using Test
include("AssemblySpace.jl")

@test add(Scalar(1),Scalar(4)) == Scalar(5.0)
@test add(Vec(Vector{Float64}([2.6])), Vec(Vector{Float64}([1.4]))).vec == Vector{Float64}([4.0])

v1 = Vec(Vector([1.4,5.7,3.2]))
v2 = Vec(Vector([1.4,5.7,3.2]))
@test add(v1, v2).vec == Vector{Float64}([2.8, 11.4, 6.4])

block = :(Scalar(5.0))
a = AssemblyPath(Vector([block]), Vector([typeof(eval(block))]))
new_expr = :(add(Scalar(5.0), Scalar(1.5)))

expected_path = Vector([:(Scalar(5.0)), :(add(Scalar(5.0), Scalar(1.5)))])
expected_types = Vector([Scalar, Scalar])

new_ap = create_assembly_path(a, new_expr)
@test (new_ap.path == expected_path) && (new_ap.types == expected_types)

# test that create_assembly_path doesn't override input path
ap_old = AssemblyPath(Expr[:(add(Scalar(1.0), Scalar(1.0)))],Type[Scalar])
new_model = :(add(add(Scalar(1.0), Scalar(1.0)), Scalar(1.0)))
ap_new = create_assembly_path(ap_old, new_model)
@test ap_old != ap_new
@test ap_old.path == Expr[:(add(Scalar(1.0), Scalar(1.0)))]
@test ap_old.types == Type[Scalar]
@test ap_new.path == Expr[:(add(Scalar(1.0), Scalar(1.0))), :(add(add(Scalar(1.0), Scalar(1.0)), Scalar(1.0)))]
@test ap_new.types == Type[Scalar, Scalar]

x = Vec([2/3, 4/3, 6/3])
model = :(multiply(add(Scalar(1.0), reciprocal(Scalar(2.0))), x))
@test eval(model).vec == [1,2,3]

add_methods = methods(add)
mult_methods = methods(multiply)
recip_methods = methods(reciprocal)
@test get_method_argument_types(add_methods[1]) == [Scalar, Scalar] || get_method_argument_types(add_methods[1]) == [Vec, Vec]
@test get_method_argument_types(add_methods[2]) == [Vec, Vec] || get_method_argument_types(add_methods[2]) == [Scalar, Scalar]
@test get_method_argument_types(mult_methods[1]) == [Scalar, Scalar] || get_method_argument_types(mult_methods[1]) == [Scalar, Vec]
@test get_method_argument_types(mult_methods[2]) == [Scalar, Vec] || get_method_argument_types(mult_methods[2]) == [Scalar, Scalar]
@test get_method_argument_types(recip_methods[1]) == [Scalar]

@test models_of_type(Scalar, building_blocks, building_block_types) == Expr[:(Scalar(0.0)),:(Scalar(1.0))]
@test models_of_type(Vec, building_blocks, building_block_types) == Expr[building_blocks[3]]

@test models_of_type(Scalar, push!(copy(building_blocks), :(Scalar(2.0))), push!(copy(building_block_types), Scalar)) == Expr[:(Scalar(0.0)),:(Scalar(1.0)),:(Scalar(2.0))]

# the order of the list returned by methods is arbitrary, so we only require that this works for methods(add)[1] or methods(add)[2]. Silly I know.
argument_matrix1 = collect(arguments_that_match_type_signature(methods(add)[1], building_blocks, building_block_types))
true_argument_matrix1 = Matrix{Tuple{Expr, Expr}}([[(:(Scalar(0.0)), :(Scalar(0.0))), (:(Scalar(1.0)), :(Scalar(0.0)))] [(:(Scalar(0.0)), :(Scalar(1.0))), (:(Scalar(1.0)), :(Scalar(1.0)))]])
argument_matrix2 = collect(arguments_that_match_type_signature(methods(add)[2], building_blocks, building_block_types))
true_argument_matrix2 = Matrix{Tuple{Expr, Expr}}([[(:(Scalar(0.0)), :(Scalar(0.0))), (:(Scalar(1.0)), :(Scalar(0.0)))] [(:(Scalar(0.0)), :(Scalar(1.0))), (:(Scalar(1.0)), :(Scalar(1.0)))]])
@test (argument_matrix1 == true_argument_matrix1) || (argument_matrix2 == true_argument_matrix2)
