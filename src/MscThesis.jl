module MscThesis
# -----------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------
# Folders
DEPENDENCY_DIR = "deps"
TEMP_DIR = tempdir()
# Indicators
# -----------------
QHV_TEMP_DIR = mktempdir(TEMP_DIR)
QHV_EXECUTABLE = "$DEPENDENCY_DIR/QHV/d"
QHV_MAX_DIM = 15

export QHV_EXECUTABLE, QHV_TEMP_DIR, QHV_MAX_DIM

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
include("./Utils.jl")

# Internals
include("./optimization/Optimization.jl")
include("./optimization/Pareto.jl")
include("./optimization/Sampling.jl")

# Solvers
include("./optimization/solvers/Platypus.jl")
include("./optimization/solvers/PlatypusSolver.jl")
include("./optimization/solvers/Metamodels.jl")
include("./optimization/solvers/MetaSolver.jl")

# Benchmarks
include("./indicators/Indicators.jl")
end
