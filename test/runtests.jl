using Test


@testset "MscThesis" begin
    # Unit test for module internals
    include("Optimization.jl")
    include("Pareto.jl")
    include("Sampling.jl")

    # Unit tests for solvers
    include("MetaModels.jl")
    include("MetaSolver.jl")


end
