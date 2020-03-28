module ADOPT
using Dates
using Random

using DelimitedFiles
using Distances
using Statistics
using LinearAlgebra

include("./Parameters.jl");
include("./Utils.jl");
include("./FileUtils.jl");
#= ---------------------------------------------------------------------- #
    Configurations
# ---------------------------------------------------------------------- =#
# Step 1. Log Level, by default we write all the logs to a file
using Logging

loglevel!(level) = let
    logger = ConsoleLogger(stdout, level);
    global_logger(logger);
    @info "[$(now())] Global logger switched to ConsoleLogger with $(level) level."
    end

loglevel!(Logging.Info)
# loglevel!(Logging.Debug)

# Step 2. Application logs - By default we write all the results of optimization processes to files
# We also log the state of every optimization run, so that users are aware of the state
results_dir = Parameter("./results")
results_file = Parameter("$(results_dir())/default-results.csv")
config_file = Parameter("$(results_dir())/default.config")

# create_temp_dir(results_dir())

# Optimization Internals

# This maps how much a solution is penalized by default
ϵ = Parameter(0.1)
include("./optimization/Optimization.jl")
include("./optimization/Pareto.jl")

# Extension of Pareto functionality to Optimization concepts
Pareto.ParetoResult(solutions::Vector{Solution}) =
    isempty(solutions) ? throw(ArgumentError("Empty set of solutions provide does not provide enough information to create ParetoResult")) :
    let nvars = nvariables(solutions[1])
        nobjs = nvobjectives(solutions[1])

        pd = Pareto.ParetoResult(nvars, nobjs)
        map(solutions) do solution
            push!(pd, Pareto.variables(solution), Pareto.objectives(solution))
        end
        pd
    end
Pareto.is_nondominated(solutions::Vector{Solution}) =
    isempty(solutions) ? true :
    let V = hcat(map(objectives, solutions)...)
        [solutions[i] for i in 1:length(solutions)
            if Pareto.is_nondominated(V[:, i], V[:,1:end .!=i])]
    end

# Benchmarks
include("./Benchmark.jl")
include("./indicators/Indicators.jl")
include("./indicators/CumulativeIndicators.jl")
export Hypervolume
#= ---------------------------------------------------------------------- #
    Solvers
# ---------------------------------------------------------------------- =#
include("./optimization/solvers/Platypus.jl")
include("./optimization/solvers/PlatypusSolver.jl")

include("./optimization/Sampling.jl")
include("./optimization/solvers/SamplingSolver.jl")
export PlatypusSolver, SamplingSolver

include("./optimization/solvers/ScikitLearnModels.jl")
include("./optimization/solvers/MetaSolver.jl")
export MetaSolver, Surrogate

algs_to_solvers = Dict()

# Design Of Experiments
map(alg -> algs_to_solvers[alg] = SamplingSolver,
    [randomMC, stratifiedMC, latinHypercube, kfactorial, boxbehnken])

# Metaheuristics (Pareto-Based Optimization)
map(alg -> algs_to_solvers[alg] = PlatypusSolver, [
    CMAES, EpsMOEA, EpsNSGAII, GDE3, IBEA, MOEAD,
    NSGAII, NSGAIII, PAES, PESA2, OMOPSO, SMPSO, SPEA2])

# Syntax
# solve(algorithm=:randomMC,
#       params=Dict(:max_evals => 100, :nondominated_only => true),
#       problem=Model(...))
solve(;algorithm, params=Dict(), problem) = let
    # Change `params` to avoid any entropy later in the call chain
    new_params = copy(params)
    evs = pop!(new_params, :max_evals, 100)
    nd_only = pop!(new_params, :nondominated_only, true)

    solve(algorithm, new_params, problem, evs, nd_only)
end

# Syntax
# solve(algorithm=:NSGAII,
#       params=Dict(:population_size => 15),
#       max_evals=50,
#       nondominated_only=false,
#       problem=Model(...))
solve(algorithm, params, problem, max_evals, nondominated_only) =
    if algorithm in keys(algs_to_solvers)
        solver = get_solver(algs_to_solvers[algorithm], algorithm, params, max_evals, nondominated_only);
        solve(solver, problem)
    else
        throw(DomainError("Algorithm $(string(algorithm)) is currently not supported..."))
    end

# Syntax
# solve(algorithm=PlatypusSolver(...),
#       max_evals=50,
#       nondominated_only=false,
#       problem=Model(...))
solve(algorithm::AbstractSolver, _, problem, max_evals, nondominated_only) = begin
     max_evaluations!(algorithm, max_evals)
     nondominated_only!(algorithm, nondominated_only)
     solve(algorithm, problem)
 end

end # Module
