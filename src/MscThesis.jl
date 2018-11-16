module MscThesis

# Submodules
# ----------

# Utils
# include("./Utils.jl")
# Internals
include("./optimization/Optimization.jl")
include("./optimization/Sampling.jl")
# Solvers
include("./optimization/solvers/Platypus.jl")
include("./optimization/solvers/PlatypusSolver.jl")
include("./optimization/solvers/SurrogateModels.jl")
include("./optimization/solvers/SurrogateSolver.jl")

end
