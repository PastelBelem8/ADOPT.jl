# Multi modal function
using JuMP
using NLopt

m = Model(solver = NLoptSolver(algorithm=:GN_CRS2_LM, maxeval=100))
# f(x) = x^4 - 4x^2
# JuMP.register(m, :f, 2, f; autodiff=true)

@variable(m, -3 <= x <= 3)
@NLobjective(m, Min,  x^4 - 4x^2)

print(m)

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
