module MscThesis

# Submodules
# ----------

using Dates
using Random

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

end
