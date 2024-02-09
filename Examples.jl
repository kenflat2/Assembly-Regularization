include("AssemblySpace.jl")

global x::Vec
global y::Vec

#=
# Lean y=x
x = Vec([-1.0, 0.0, 1.0])
y = Vec([-2.0, 0.0, 1.0])
λ = 0.1
assemble(λ, 10)
# Solution: x

# Learn 4x
x = Vec(randn!(zeros(10)))
y = Vec(4 * x.vec)
assemble(λ, 10)
# Solution: +(+(x,x),+(x,x))

# Learn 4x + noise
sigma = 0.5
noise = randn!(zeros(10)) * sigma
x = Vec(randn!(zeros(10)))
y = Vec(4 * x.vec + noise)
assemble(λ, 10)
# Solution: 4x

# Learn πx + noise
sigma = 0.1
noise = randn!(zeros(10)) * sigma
x = Vec(randn!(zeros(10)))
y = Vec(π * x.vec + noise)
assemble(λ, 10)
# Solution: 3x

# Learn πx + noise, where the variance is  much higher than that of the model.
sigma = 4
λ = 1
noise = randn!(zeros(10)) * sigma
x = Vec(randn!(zeros(10)))
y = Vec(x.vec + noise)
assemble(λ, 10)
# Solution: 1
# The true mean of y is 1.15


# x = 1/2
λ = 0.1
L = 100
x = Vec(randn!(zeros(L)))
y = Vec(ones(L)*1/2)
assemble(λ, 10)
# Solution y = 1/2

# y = 3/4x
λ = 0.01
L = 100
x = Vec(randn!(zeros(L)))
y = Vec(x.vec * 3/4)
model = assemble(λ, 100)
# Solution: x
# This result is problematic, the true best model has an assembly index of 6 and is 3/4x
# There is no reason for the algorithm to search through the set of scalars because it time it does.
# That said, the domain is roughly [-1,1] so the distinction in slopes is not vast.
=#

# y = 3/4x over x ~ N(0, 10 )
λ = 0.1
L = 10
x = Vec(randn!(zeros(L)) * 10) 
y = Vec(x.vec * 3/4)
model = assemble(λ, 10)
# This one is taking forever...
# and it returns x after 114 generations. Clearly, we need to explore
# the search space in a more wide way for this method to succeed.