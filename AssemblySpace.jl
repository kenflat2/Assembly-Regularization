abstract type Model end

abstract type BuildingBlock <: Model end

abstract type Operation <: Model end

struct bank 
    models::[Model]
end

function init_banks()
    
end

struct scalar0 <: Model
    val::Int
end

struct scalar1 <: Model
    val::Int
end

struct operation <: Model
    op::
end

type operation 

function eval(m::Model, x::Real)
    if m isa scalar0
        return 0
    elif m isa scalar1
        return 1
    elif m isa operation
        return 
    return eval(m).(X)