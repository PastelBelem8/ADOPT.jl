using Main.ADOPT
using Main.ADOPT.Platypus
# -------------------------------------------------------------------------
# Step 1. Define the problem(s)
# -------------------------------------------------------------------------
# Here, we define two MOO problems
# (https://en.wikipedia.org/wiki/Test_functions_for_optimization):
# - schaffer1
# - kursawe
# - binhkorn
# -------------------------------------------------------------------------

# # Schaffer function nº 1 (Unconstrained)
# schaffer1_vars = [RealVariable(-10, 10)]
#
# schaffer1_f1(x) = x[1]^2
# schaffer1_f2(x) = (x[1] - 2)^2
# schaffer1_objs = [Objective(schaffer1_f1), Objective(schaffer1_f2)]
#
# schaffer1 = Model(schaffer1_vars, schaffer1_objs)
# # # Kursawe function (Unconstrained)
# kursawe_vars = [RealVariable(-5, 5), RealVariable(-5, 5), RealVariable(-5, 5)]
# kursawe_f1(x) = sum([-10 * exp(-0.2 * √(x[i]^2 + x[i+1]^2)) for i in 1:2])
# kursawe_f2(x) = sum([abs(x[i])^0.8 + 5sin(x[i]^3) for i in 1:3])
# #
# kursawe_objs = [Objective(kursawe_f1), Objective(kursawe_f2)]
# kursawe = Model(kursawe_vars, kursawe_objs)

# ----------------------------------------------------------------------
# Binh and Korn function (Constrained)
# ----------------------------------------------------------------------
binhkorn_vars = [RealVariable(0, 5), RealVariable(0, 3)]

binhkorn_f1(x) = 4x[1]^2 + 4x[2]^2
binhkorn_f2(x) = (x[1] - 5)^2 + (x[2] - 5)^2
binhkorn_objs = [Objective(binhkorn_f1), Objective(binhkorn_f2)]

binhkorn_c1(x) = (x[1] - 5)^2 + x[2]^2 - 25 # <= 0
binhkorn_c2(x) = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7 # >= 0
binhkorn_cnstrs = [Constraint(binhkorn_c1, <=), Constraint(binhkorn_c2, >=)]

binhkorn = Model(binhkorn_vars, binhkorn_objs, binhkorn_cnstrs)

maxevals = 300
nparticles = 5
generator = Dict(:name => InjectedPopulation,
                 :solutions => [[1, 0], [1, 1], [1, 2], [1, 2.5], [1, 0.3]],
                 :problem => binhkorn
            )
PSO_params = Dict(
    :leader_size => nparticles,
    :swarm_size => nparticles,
    :generator => generator
)
EA_params = Dict(
    :population_size => nparticles,
    :generator => generator
)
Main.ADOPT.benchmark(
    nruns=2,
    algorithms=[(SMPSO, PSO_params), (SPEA2, EA_params)],
    problem=binhkorn,
    max_evals=maxevals
)

# ----------------------------------------------------------------------


# -------------------------------------------------------------------------
# Evolutionary Solver
# -------------------------------------------------------------------------
# generator = Dict(:name => InjectedPopulation,
#                  :solutions => [[-10], [-5], [0], [5], [10]],
#                  :problem => schaffer1)
#
# gen = convert(Main.ADOPT.Platypus.InjectedPopulation, generator)
# gen.pyo.solutions
#
# ea_params = Dict(:population_size => 5,
#                  :generator => generator)
#
# # Platypus Solver
# solver = Main.ADOPT.PlatypusSolver(NSGAII,
#                 max_eval=20,
#                 algorithm_params=ea_params,
#                 nondominated_only=true)
#
# schaffer1_sols = solve(solver, schaffer1)
# # binhkorn_sols = solve(solver, binhkorn)
# # kursawe_sols = solve(solver, kursawe)
#
# PSOs_params = Dict( :leader_size => 5, :swarm_size => 5, :max_iterations => 5 * 2, :epsilons => [0.1, 0.1], :generator => generator)
# solver2 =  Main.ADOPT.PlatypusSolver(OMOPSO, max_eval=10, algorithm_params=PSOs_params, nondominated_only=false)
# # modified_kursawe = Model([IntVariable(-5, 5), IntVariable(-5, 5), IntVariable(-5, 5)], [Objective(kursawe_f1), Objective(kursawe_f2)])
# # modified_kursawe_sols = solve(solver2, modified_kursawe)
#
# schaffer1_sols = solve(solver2, schaffer1)
# # binhkorn_sols = solve(solver2, binhkorn)
# # kursawe_sols = solve(solver2, kursawe)
# # -------------------------------------------------------------------------
# Sampling Solver
# -------------------------------------------------------------------------
#=
using Main.MscThesis.Sampling
# Sampling Solver
sampling_params = Dict(:sampling_function => randomMC)
solver = SamplingSolver(;algorithm_params=sampling_params,
                        max_eval=500,
                        nondominated_only=true)

schaffer1_sols = solve(solver, schaffer1)
binhkorn_sols = solve(solver, binhkorn)
kursawe_sols = solve(solver, kursawe)
=#

# -------------------------------------------------------------------------
# Meta Solver
# -------------------------------------------------------------------------
#=
using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
using Main.MscThesis.ScikitLearnModels: sk_fit!, sk_predict, SVR, GaussianProcessRegressor, LinearRegression

sklearn_model = SVR() # GaussianProcessRegressor() # LinearRegression
surrogate = Surrogate(  sklearn_model,
                        objectives=schaffer1_objs,
                        creation_f=sk_fit!,
                        update_f=sk_fit!,
                        evaluation_f=sk_predict)
meta_params = Dict(:sampling_function => randomMC, :nsamples => 30)

# Sampling Solver
optimiser1 = SamplingSolver(;algorithm_params=Dict(:sampling_function => latinHypercube), max_eval=200, nondominated_only=false)

# Platypus Solver
optimiser2 = Main.MscThesis.PlatypusSolver(NSGAII, max_eval=500, algorithm_params=Dict(:population_size => 50), nondominated_only=true)

solver = Main.MscThesis.MetaSolver(optimiser2; surrogates=[surrogate], max_eval=200,
                sampling_params=meta_params, nondominated_only=true)

schaffer1_sols = solve(solver, schaffer1)
binhkorn_sols = solve(solver, binhkorn)
kursawe_sols = solve(solver, kursawe)
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
myplot(kursawe_sols)
=#
