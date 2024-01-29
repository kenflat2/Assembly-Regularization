abstract type Model end
abstract type Scalar <: Model end
abstract type Operation <: Model end
abstract type Vec <: Model end

struct Scalar <: Model end

struct Zero <: Scalar end
struct One <: Scalar end

struct Data <: Vec
    sym::Symbol
end

# function get_val(a::BuildingBlock) end
# get_val(a::Zero) = 0.0::Float64
# get_val(a::One) = 1.0::Float64
# get_val(a::Data) = eval(a.sym)

function add(a1::Model, a2::Model) <: Operation end
add(a1::Zero, a2::Zero) = :(0.0)
add(a1::Zero, a2::One) = :(1.0)
add(a1::One, a2::Zero) = :(1.0)
add(a1::One, a2::One) = :(2.0)
add(v1::Data, v2::Data) = :(v1 + v2)

function multiply(a1::Model, a2::Model) <: Operation end
multiply(a1::Scalar, a2::Vector) = get_val(a1) * get_val(a2)

function reciprocal(a::Scalar) <: Operation end
reciprocal(a::One) = 1
reciprocal(a::Scalar) = 1 / get_val(a)

function getconcretetypes(t::Type)
    if isconcretetype(t)
        return Vector{Type}([t])
    end

    types = Vector{Type}()
    for subtype in subtypes(t)
        append!(types, getconcretetypes(subtype))
    end

    return types
end