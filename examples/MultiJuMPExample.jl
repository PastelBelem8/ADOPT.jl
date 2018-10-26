using MultiJuMP, JuMP
using Ipopt

m = MultiModel(solver = IpoptSolver())

@variable(m, x[i=1:5])
@NLexpression(m, f1, sum(x[i]^2 for i=1:5))
@NLexpression(m, f2, 3x[1]+2x[2]-x[3]/3+0.01*(x[4]-x[5])^3)
@NLconstraint(m, x[1]+2x[2]-x[3]-0.5x[4]+x[5]==2)
@NLconstraint(m, 4x[1]-2x[2]+0.8x[3]+0.6x[4]+0.5x[5]^2 == 0)
@NLconstraint(m, sum(x[i]^2 for i=1:5) <= 10)

iv1 = [0.3, 0.5, -0.26, -0.13, 0.28] # Initial guess
obj1 = SingleObjective(f1, sense = :Min,
                       iv = Dict{Symbol,Any}(:x => iv1))
obj2 = SingleObjective(f2, sense = :Min)

md = getMultiData(m)
md.objectives = [obj1, obj2]
md.pointsperdim = 20
solve(m, method = :NBI) # method = :WS or method = :EPS

# DTLZ1 Problem
using JuMP, MultiJuMP
using NLopt

m = MultiModel(solver = NLoptSolver(algorithm=:GN_CRS2_LM, maxeval=100))
f(z...) = 2 + sum([(z[i]-0.5)^2 - cos(20 * 3.14 *(z[i]-0.5)) for i in 1:length(z)])
JuMP.register(m, :f, 2, f; autodiff=true)

@variable(m, 0 <= x <= 1)
@variable(m, 0 <= y <= 1)
@NLexpression(m, f2, 0.5*(1-x)*(1+f(x, y)))
@NLexpression(m, f1, 0.5*x*(1+f(x, y)))

obj1 = SingleObjective(f1, sense = :Min)
obj2 = SingleObjective(f2, sense = :Min)

md = getMultiData(m)
md.objectives = [obj1,obj2]
md.pointsperdim = 50
solve(m, method = :NBI) #

using Plots
pltnbi = plot(md)

results = getMultiData(m)
results.paretofront
