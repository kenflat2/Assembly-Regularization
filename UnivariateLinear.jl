# In this file we specify what a vector space is. We will assemble 
# different vectors and therefore models within the vector space
# framework.

# To allow for assembly, you must specify
#   Building Blocks - a set of expression you wish to assemble to
#                     produce models
#   Operations - functions which take combine one or two models to 
#                produce another model
# The model inputs should be left as a symbol in this page, then 
# later set to be an object of an allowed model type. For instance,
# data are entered into this model as a vector.
struct Scalar
    val::Float64
end;

struct Vec
    vec::Vector{Float64}
end;

function add(a1::Scalar, a2::Scalar)
    return Scalar(a1.val + a2.val)
end;

function add(v1::Vec, v2::Vec)
    return Vec(v1.vec + v2.vec)
end;

function multiply(a1::Scalar, a2::Scalar)
    return Scalar(a1.val * a2.val)
end

function multiply(a::Scalar, v2::Vec) 
    return Vec(a.val * v2.vec)
end

function reciprocal(a::Scalar)
	return if a.val == 0
        return Scalar(0) 
    else 
        return Scalar(1 / a.val) 
    end
end;

building_blocks = Vector{Expr}([:(Scalar(0.0)), :(Scalar(1.0)), quote x end]);
building_block_types = Vector{Type}([Scalar, Scalar, Vec]);
operations = Vector([add, multiply, reciprocal]);