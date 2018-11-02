module MscThesis

# Submodules
# ----------

# Internals
include("./optimization/Optimization.jl")
include("./optimization/Sampling.jl")
# Solvers
include("./optimization/Platypus.jl")
include("./optimization/PlatypusSolver.jl")

end
