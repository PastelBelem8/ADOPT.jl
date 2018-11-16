using Test


@testset "MscThesis" begin
    # Unit test for module internals
    include("Optimization.jl")
    include("Sampling.jl")

    # Unit tests for solvers
    include("SurrogateModels.jl")


end
