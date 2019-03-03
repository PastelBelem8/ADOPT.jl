module MscThesis
# -----------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------
# Folders
# DEPENDENCY_DIR = "deps"
# TEMP_DIR = tempdir()
# # Indicators
# # -----------------
# QHV_TEMP_DIR = mktempdir(TEMP_DIR)
# QHV_EXECUTABLE = "$DEPENDENCY_DIR/QHV/d"
# QHV_MAX_DIM = 15
#
# export QHV_EXECUTABLE, QHV_TEMP_DIR, QHV_MAX_DIM

# Dependencies
# -------------
using Dates
using Random

using Distances
using LinearAlgebra
using Statistics

# Submodules
# -------------

# Utils
include("./Parameters.jl")
include("./Utils.jl")

# Internals
include("./optimization/Optimization.jl")
include("./optimization/Pareto.jl")

# -----------------------------------------------------------------------
# Extension of Pareto functionality to Optimization concepts
# -----------------------------------------------------------------------
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

# Solvers
include("./optimization/solvers/Platypus.jl")
include("./optimization/solvers/PlatypusSolver.jl")

include("./optimization/Sampling.jl")
include("./optimization/solvers/SamplingSolver.jl")
export PlatypusSolver, SamplingSolver

include("./optimization/solvers/ScikitLearnModels.jl")
include("./optimization/solvers/MetaSolver.jl")

# Benchmarks
# include("./indicators/Indicators.jl")
end
