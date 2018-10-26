# DTLZ1 Problem - Multi modal function
using JuMP
using Ipopt

m = Model(solver = NLoptSolver(algorithm=:GN_CRS2_LM, maxeval=100))
a(z...) = 2 + sum([(z[i]-0.5)^2 - cos(20 * 3.14 *(z[i]-0.5)) for i in 1:length(z)])
JuMP.register(m, :a, 2, a; autodiff=true)

@variable(m, 0 <= x <= 1)
@variable(m, 0 <= y <= 1)
# @NLexpression(m, f1, 0.5*x*(1+a(x, y)))
# @NLexpression(m, f2, 0.5*(1-x)*(1+a(x, y)))

@NLobjective(m, Min, 0.5*x*(1+a(x, y)))

print(m)

solve(m) #

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))
