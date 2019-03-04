module MscThesis
using Dates
using Random

using Distances
using Statistics
using LinearAlgebra


include("./Parameters.jl")
include("./Utils.jl")
include("./FileUtils.jl")
#= ---------------------------------------------------------------------- #
    Configurations
# ---------------------------------------------------------------------- =#
# Step 1. Log Level, by default we write all the logs to a file
using Logging

log_level = Logging.Info
# log_level = Logging.Debug
logger = ConsoleLogger(stdout, log_level); global_logger(logger);
@info "[$(now())] Starting logger with $(log_level) level."

# TODO - Log to file and console (check other frameworks, e.g., Memento, MiniLogging, Micrologging)
Logging.min_enabled_level(global_logger())


# Step 2. Application logs - By default we write all the results of optimization processes to files
# We also log the state of every optimization run, so that users are aware of the state
results_dir = Parameter("./results")
results_file = Parameter("$(results_dir())/default-results.csv")
config_file = Parameter("$(results_dir())/default.config")

try
    mkdir(results_dir())
    @info "[$(now())] Creating directory $(results_dir()) to place the results of subsequent optimization runs..."
catch e
    if isa(e, SystemError)
        @info "[$(now())] Results directory $(results_dir()) already exists. Optimization output files will be placed in that directory."
    end
end

#= ---------------------------------------------------------------------- #
    Utils
# ---------------------------------------------------------------------- =#

# Utils

# Optimization Internals
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
# include("./indicators/Indicators.jl")

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

end
