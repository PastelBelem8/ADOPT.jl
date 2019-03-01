using Main.MscThesis

# -------------------------------------------------------------------------
# Step 1. Define the problem(s)
# -------------------------------------------------------------------------
# Here, we define two MOO problems
# (https://en.wikipedia.org/wiki/Test_functions_for_optimization):
# - schaffer1
# - binhkorn
# -------------------------------------------------------------------------

# Schaffer function nº 1 (Unconstrained)
schaffer1_vars = [RealVariable(-10, 10)]

schaffer1_f1(x) = x[1]^2
schaffer1_f2(x) = (x[1] - 2)^2
schaffer1_objs = [Objective(schaffer1_f1), Objective(schaffer1_f2)]

schaffer1 = Model(schaffer1_vars, schaffer1_objs)

# Binh and Korn function (Constrained)
binhkorn_vars = [RealVariable(0, 5), RealVariable(0, 3)]

binhkorn_f1(x) = 4x[1]^2 + 4x[2]^2
binhkorn_f2(x) = (x[1] - 5)^2 + (x[2] - 5)^2
binhkorn_objs = [Objective(binhkorn_f1), Objective(binhkorn_f2)]

binhkorn_c1(x) = (x[1] - 5)^2 + x[2]^2 - 25 # <= 0
binhkorn_c2(x) = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7 # >= 0
binhkorn_cnstrs = [Constraint(binhkorn_c1, <=), Constraint(binhkorn_c2, >=)]

binhkorn = Model(binhkorn_vars, binhkorn_objs, binhkorn_cnstrs)

# -------------------------------------------------------------------------
# Evolutionary Solver
# -------------------------------------------------------------------------
#=
using Main.MscThesis.Platypus

ea_params = Dict(:population_size => 100)
# Platypus Solver
solver = Main.MscThesis.PlatypusSolver(NSGAII,
                max_eval=200,
                algorithm_params=ea_params,
                nondominated_only=true)

schaffer1_sols = solve(solver, schaffer1)
binhkorn_sols = solve(solver, binhkorn)
=#

# -------------------------------------------------------------------------
# Sampling Solver
# -------------------------------------------------------------------------
#=
using Main.MscThesis.Sampling
# Sampling Solver
sampling_params = Dict(:sampling_function => randomMC)
solver = SamplingSolver(;algorithm_params=sampling_params,
                        max_eval=300,
                        nondominated_only=true)

schaffer1_sols = solve(solver, schaffer1)
binhkorn_sols = solve(solver, binhkorn)
=#
# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
#=
using Plots

myplot(solutions::Vector{Solution}) = let
    objs = cat(map(objectives, solutions)...; dims=2)
    plot(objs[1,:], objs[2,:], seriestype=:scatter)
end

myplot(schaffer1_sols)
myplot(binhkorn_sols)
=#
