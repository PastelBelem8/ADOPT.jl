<<<<<<< HEAD
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
=======
# using Logging
# # io = open("log.txt", "w+")
# # logger = SimpleLogger(io)
# # global_logger(logger)

include("configurations.jl")
include("utils.jl")
include("./indicators/MOOIndicators.jl")
>>>>>>> metrics
