using Test


@testset "ADOPT-unit-tests" begin
    print("Starting Optimization Tests")
    include("unit/Optimization.jl")

    print("Starting Pareto Tests")
    include("unit/Pareto.jl")

    print("Starting Sampling Tests")
    include("unit/Sampling.jl")

    print("Starting MetaModels Tests")
    include("unit/MetaModels.jl")

    print("Starting MetaSolvers Tests")
    include("unit/MetaSolver.jl")
end

@testset "ADOPT-integration-tests" begin
    # Test integration

end
