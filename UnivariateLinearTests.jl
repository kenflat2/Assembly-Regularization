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

x = Vec([2/3, 4/3, 6/3])
model = :(multiply(add(Scalar(1.0), reciprocal(Scalar(2.0))), x))
@test eval(model).vec == [1,2,3]

argument_matrix = collect(arguments_that_match_type_signature(methods(add)[1], building_blocks, building_block_types))
true_argument_matrix = Matrix{Tuple{Union{Expr, Symbol}, Union{Expr, Symbol}}}([[(:(Scalar(0.0)), :(Scalar(0.0))), (:(Scalar(1.0)), :(Scalar(0.0)))] [(:(Scalar(0.0)), :(Scalar(1.0))), (:(Scalar(1.0)), :(Scalar(1.0)))]])
@test argument_matrix == true_argument_matrix