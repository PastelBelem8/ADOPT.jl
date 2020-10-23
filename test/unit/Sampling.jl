module SamplingTests

using Test

import MscThesis.Sampling

# Tests -----------------------------------------------------------------
@testset "Auxiliar functions Tests" begin

    @testset "set_seed! Tests" begin
        Sampling.set_seed!(0)
        @test rand() == 0.8236475079774124
        @test rand() == 0.9103565379264364
        @test rand() == 0.16456579813368521
        @test [rand() for _ in 1:3] == [0.17732884646626457, 0.278880109331201, 0.20347655804192266]
        Sampling.set_seed!(1)
        @test rand() != 0.8236475079774124
        Sampling.set_seed!(0)
        @test rand() + rand() == 1.7340040459038488
        @test_throws DomainError Sampling.set_seed!(-1)
    end

    @testset "rand_range Tests" begin
        f(m1, m2, s=0) = begin Sampling.set_seed!(s); Sampling.rand_range(m1, m2) end

        @test f(0, 0) == f(0, 0) == 0
        @test f(0, 1) == f(0, 1)
        @test f(1, 1) == 1

        @test f(0, 1) == -1*f(0, -1)
        @test count(x -> -1 <= x <= 1, [Sampling.rand_range(-1, 1) for _ in 1:10000]) == 10000

        @test f(0, 1) ==  0.8236475079774124
        @test f(70, 71) ==  70.8236475079774124
        @test f(79, 90) ==  88.06012258775154
        @test f(0, 1) + f(0, 1) == f(0, 2)
    end
end

@testset "Monte Carlo Sampling Tests" begin
     ss(s) = Sampling.set_seed!(s)
     seq(n) = begin ss(0); [rand() for _ in 1:n] end

     @testset "random_sample Tests" begin

          @test begin s = Sampling.random_sample(0); isa(s, Vector) && isempty(s) end
          @test_throws DomainError Sampling.random_sample(-1)

          @test begin ss(0); Sampling.random_sample(1) ==[0.8236475079774124] end
          @test begin ss(0); Sampling.random_sample(10) == seq(10) end
          @test begin ss(1); Sampling.random_sample(1000) != seq(1000) end

     end
     @testset "random_samples Tests" begin
          ss(0)
          @test begin s = Sampling.random_samples(0, 0); isa(s, Matrix) && isempty(s) end
          @test begin s = Sampling.random_samples(10, 0); isa(s, Matrix) && isempty(s) end

          @test_throws DomainError Sampling.random_samples(-1, 0)
          @test_throws DomainError Sampling.random_samples(-1, 1)
          @test_throws DomainError Sampling.random_samples(0, -1)
          @test_throws DomainError Sampling.random_samples(1, -1)

          @test begin ss(0); Sampling.random_samples(2, 10)[:] == seq(20) end
          @test begin ss(0); Sampling.random_samples(2, 10) == reshape(seq(20), (2,10)) end
     end

     @testset "randomMC Tests" begin
          ss(0)
          @test begin s = Sampling.randomMC(0, 0); isa(s, Matrix) && isempty(s) end
          @test begin s = Sampling.randomMC(10, 0); isa(s, Matrix) && isempty(s) end

          @test_throws DomainError Sampling.randomMC(-1, 0)
          @test_throws DomainError Sampling.randomMC(-1, 1)
          @test_throws DomainError Sampling.randomMC(0, -1)
          @test_throws DomainError Sampling.randomMC(1, -1)

          @test begin ss(0); Sampling.randomMC(2, 10)[:] == seq(20) end
          @test begin ss(0); Sampling.randomMC(2, 10) == reshape(seq(20), (2,10)) end
     end

     @testset "stratifiedMC Tests" begin
          @test begin ss(0); s = Sampling.stratifiedMC(0, Vector{Int}()); isa(s, Matrix) && isempty(s) end
          @test begin ss(0); s = Sampling.stratifiedMC(1, [1]); isa(s, Matrix) && s ==  reshape(seq(1), (1, 1)) end
          @test begin ss(0); s = Sampling.stratifiedMC(2, [1, 1]); isa(s, Matrix) && s == reshape(seq(2), (2,1)) end
          @test begin ss(0); s = Sampling.stratifiedMC(2, [1; 1]); isa(s, Matrix) && s[:] == seq(2) end

          # Method Errors (due to type mismatch)
          @test_throws MethodError Sampling.stratifiedMC(0, 0)
          @test_throws MethodError Sampling.stratifiedMC(1, 2)
          @test_throws MethodError Sampling.stratifiedMC(2, 2)
          @test_throws MethodError Sampling.stratifiedMC(1, 1.0)
          @test_throws MethodError Sampling.stratifiedMC(1.0, 1)
          @test_throws MethodError Sampling.stratifiedMC(0, -1)
          @test_throws MethodError Sampling.stratifiedMC(1, [])
          @test_throws MethodError Sampling.stratifiedMC(1, [1.0])
          @test_throws MethodError Sampling.stratifiedMC(1, [[1]])
          @test_throws MethodError Sampling.stratifiedMC(4, [1 2; 3 4])

          # Domain errors
          @test_throws DomainError Sampling.stratifiedMC(-1,[0])
          @test_throws DomainError Sampling.stratifiedMC(-1,[1])
          @test_throws DomainError Sampling.stratifiedMC(1, [-1])
          @test_throws DomainError Sampling.stratifiedMC(2, [1, -1])
          @test_throws DomainError Sampling.stratifiedMC(3, [1, 0, 1])
          @test_throws DomainError Sampling.stratifiedMC(3, [1, -1, 1])

          # Dimension Mismatch errors
          @test_throws DimensionMismatch Sampling.stratifiedMC(1, Vector{Int}())
          @test_throws DimensionMismatch Sampling.stratifiedMC(1, [1, 2])
          @test_throws DimensionMismatch Sampling.stratifiedMC(2, [1, 2, 3])

          # Successfull tests
          s1 = seq(4); s1[2] = s1[2] * 0.5 + 0; s1[4] = s1[4] * 0.5 + 0.5;
          s2 = reshape(seq(8), (2, 4));
          s2[1, [1,3]] = s2[1, [1,3]] .* 0.5 .+ 0; s2[1, [2,4]] = s2[1, [2,4]] .* 0.5 .+ 0.5;
          s2[2, [1,2]] = s2[2, [1,2]] .* 0.5 .+ 0; s2[2, [3,4]] = s2[2, [3,4]] .* 0.5 .+ 0.5;
          @test begin ss(0); Sampling.stratifiedMC(2, [1, 2]) == reshape(s1, (2, 2)) end
          @test begin ss(0); Sampling.stratifiedMC(2, [2, 2]) == s2 end
          @test begin ss(0); size(Sampling.stratifiedMC(2, [4, 2])) == (2, 8) end
          @test begin ss(0); size(Sampling.stratifiedMC(2, [4, 3])) == (2, 12) end
 end
end

@testset "Latin Hypercube Sampling Tests" begin
     ss(s) = Sampling.set_seed!(s)
     seq(n) = begin ss(0); [rand() for i in 1:(n*2)] end

     # Method Errors (due to type mismatch)
     @test_throws MethodError Sampling.latinHypercube(2, 2, 1)
     @test_throws MethodError Sampling.latinHypercube(0, [])
     @test_throws MethodError Sampling.latinHypercube(0, Vector{Int}())
     @test_throws MethodError Sampling.latinHypercube(1, 2.0)
     @test_throws MethodError Sampling.latinHypercube(2.0, 2)
     @test_throws MethodError Sampling.latinHypercube(2.0, 1.0)

     # Domain errors
     @test_throws DomainError Sampling.latinHypercube(-1, 0)
     @test_throws DomainError Sampling.latinHypercube(-1, 1)
     @test_throws DomainError Sampling.latinHypercube(0, -1)
     @test_throws DomainError Sampling.latinHypercube(2, -1)

     # Successfull tests
     @test begin ss(1); s = Sampling.latinHypercube(0, 0); isa(s, Matrix) && isempty(s) end
     @test begin ss(0); s = Sampling.latinHypercube(1, 1); isa(s, Matrix) && s == reshape(seq(1)[2:end], (1, 1)) end
     @test begin ss(0); s = Sampling.latinHypercube(2, 1); s == reshape(seq(2)[3:end], (2, 1)) end
     @test begin ss(0); s = Sampling.latinHypercube(2, 2); all(s[:, 1] .> 0.5) && all(s[:, 2] .< 0.5) end

     # Columns must be in different bins row-wise
     @test begin ss(0); s = Sampling.latinHypercube(4, 4); all(mapslices(allunique, floor.(s .* 4 .+ 1), dims=2)) end
     @test begin ss(0); s = Sampling.latinHypercube(4, 10); all(mapslices(allunique, floor.(s .* 10 .+ 1), dims=2)) end
     @test begin ss(0); s = Sampling.latinHypercube(20, 20); all(mapslices(allunique, floor.(s .* 20 .+ 1), dims=2)) end
end

@testset "Full Factorial Sampling Tests" begin

     # Method Errors (due to type mismatch)
     @test_throws MethodError Sampling.fullfactorial(2, 2, 1)
     @test_throws MethodError Sampling.fullfactorial(0, [])
     @test_throws MethodError Sampling.fullfactorial(0, Vector{Int}())
     @test_throws MethodError Sampling.fullfactorial(1, 2.0)
     @test_throws MethodError Sampling.fullfactorial(2.0, 2)
     @test_throws MethodError Sampling.fullfactorial(2.0, 1.0)
     @test_throws MethodError Sampling.fullfactorial([], 2)

     # Domain errors
     @test_throws DomainError Sampling.fullfactorial(0, 0)
     @test_throws DomainError Sampling.fullfactorial(3, 0)
     @test_throws DomainError Sampling.fullfactorial(-1, 0)
     @test_throws DomainError Sampling.fullfactorial(-1, 1)
     @test_throws DomainError Sampling.fullfactorial(0, -1)
     @test_throws DomainError Sampling.fullfactorial(2, -1)
     @test_throws DomainError Sampling.fullfactorial(-1, -1)

     # Successfull tests
     @test begin s = Sampling.fullfactorial(0); isa(s, Matrix) && isempty(s) end

     @testset "Level 2 Full Factorial Tests" begin
          @test Sampling.fullfactorial(1) == reshape([0, 1], (1,2))
          @test Sampling.fullfactorial(2) == Sampling.fullfactorial(2, 2)
          @test Sampling.fullfactorial(2) == [0 1 0 1 ;
                                              0 0 1 1 ]
          @test Sampling.fullfactorial(3) == [0 1 0 1 0 1 0 1 ;
                                              0 0 1 1 0 0 1 1 ;
                                              0 0 0 0 1 1 1 1 ]
     end

     @testset "Level 2+ Full Factorial Tests" begin
     # Columns must be in different bins row-wise
          @test Sampling.fullfactorial(1, 3) == [ 0.0  0.5  1.0 ]
          @test Sampling.fullfactorial(1, 5) == [ 0.0  0.25  0.5  0.75  1.0 ]
          @test Sampling.fullfactorial(2, 3) == [ 0.0  0.5  1.0  0.0  0.5  1.0  0.0  0.5  1.0 ;
                                                  0.0  0.0  0.0  0.5  0.5  0.5  1.0  1.0  1.0 ]
     end
end

@testset "Box Behnken Sampling Tests" begin
     # Method Errors (due to type mismatch)
     @test_throws MethodError Sampling.boxbehnken(2.0)
     @test_throws MethodError Sampling.boxbehnken([])
     @test_throws MethodError Sampling.boxbehnken(Vector{Int}())

     # Domain errors
     @test_throws DomainError Sampling.boxbehnken(-1)
     @test_throws DomainError Sampling.boxbehnken(0)
     @test_throws DomainError Sampling.boxbehnken(1)
     @test_throws DomainError Sampling.boxbehnken(2)

     # Successfull tests
     # This section relies on pyDOE2's validation
     @test begin s = Sampling.boxbehnken(3); size(s) == (3, 13) && s == [ 0.0  0.0  0.5 ;
                                                                         1.0  0.0  0.5 ;
                                                                         0.0  1.0  0.5 ;
                                                                         1.0  1.0  0.5 ;
                                                                         0.0  0.5  0.0 ;
                                                                         1.0  0.5  0.0 ;
                                                                         0.0  0.5  1.0 ;
                                                                         1.0  0.5  1.0 ;
                                                                         0.5  0.0  0.0 ;
                                                                         0.5  1.0  0.0 ;
                                                                         0.5  0.0  1.0 ;
                                                                         0.5  1.0  1.0 ;
                                                                         0.5  0.5  0.5 ]' end
     @test begin s = Sampling.boxbehnken(4); size(s) == (4, 25) &&
                 s == [0 0 0.5 0.5;
                       1 0 0.5 0.5;
                       0 1 0.5 0.5;
                       1 1 0.5 0.5;
                       0 0.5 0 0.5;
                       1 0.5 0 0.5;
                       0 0.5 1 0.5;
                       1 0.5 1 0.5;
                       0 0.5 0.5 0;
                       1 0.5 0.5 0;
                       0 0.5 0.5 1;
                       1 0.5 0.5 1;
                       0.5 0 0 0.5;
                       0.5 1 0 0.5;
                       0.5 0 1 0.5;
                       0.5 1 1 0.5;
                       0.5 0 0.5 0;
                       0.5 1 0.5 0;
                       0.5 0 0.5 1;
                       0.5 1 0.5 1;
                       0.5 0.5 0 0;
                       0.5 0.5 1 0;
                       0.5 0.5 0 1;
                       0.5 0.5 1 1;
                       0.5 0.5 0.5 0.5]' end
     @test begin s = Sampling.boxbehnken(5); size(s) == (5, 41) &&
                 s == [0 0 0.5 0.5 0.5 ;
                       1 0 0.5 0.5 0.5 ;
                       0 1 0.5 0.5 0.5 ;
                       1 1 0.5 0.5 0.5 ;
                       0 0.5 0 0.5 0.5 ;
                       1 0.5 0 0.5 0.5 ;
                       0 0.5 1 0.5 0.5 ;
                       1 0.5 1 0.5 0.5 ;
                       0 0.5 0.5 0 0.5 ;
                       1 0.5 0.5 0 0.5 ;
                       0 0.5 0.5 1 0.5 ;
                       1 0.5 0.5 1 0.5 ;
                       0 0.5 0.5 0.5 0 ;
                       1 0.5 0.5 0.5 0 ;
                       0 0.5 0.5 0.5 1 ;
                       1 0.5 0.5 0.5 1 ;
                       0.5 0 0 0.5 0.5 ;
                       0.5 1 0 0.5 0.5 ;
                       0.5 0 1 0.5 0.5 ;
                       0.5 1 1 0.5 0.5 ;
                       0.5 0 0.5 0 0.5 ;
                       0.5 1 0.5 0 0.5 ;
                       0.5 0 0.5 1 0.5 ;
                       0.5 1 0.5 1 0.5 ;
                       0.5 0 0.5 0.5 0 ;
                       0.5 1 0.5 0.5 0 ;
                       0.5 0 0.5 0.5 1 ;
                       0.5 1 0.5 0.5 1 ;
                       0.5 0.5 0 0 0.5 ;
                       0.5 0.5 1 0 0.5 ;
                       0.5 0.5 0 1 0.5 ;
                       0.5 0.5 1 1 0.5 ;
                       0.5 0.5 0 0.5 0 ;
                       0.5 0.5 1 0.5 0 ;
                       0.5 0.5 0 0.5 1 ;
                       0.5 0.5 1 0.5 1 ;
                       0.5 0.5 0.5 0 0 ;
                       0.5 0.5 0.5 1 0 ;
                       0.5 0.5 0.5 0 1 ;
                       0.5 0.5 0.5 1 1 ;
                       0.5 0.5 0.5 0.5 0.5]';
     end
end

end # module
