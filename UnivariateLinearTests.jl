using Test
include("UnivariateLinear.jl")

@test get_val(Zero()) == 0.0
@test get_val(One()) == 1.0
@test add(One(), One()) == Sca(2.0)
@test eval(:(add(One(), Sca(7.0)))) == Sca(8.0)
@test getconcretetypes(BuildingBlock) == Vector{Type}([One,Zero,Data])
@test getconcretetypes(Model) == Vector{Type}([One,Zero,Data])