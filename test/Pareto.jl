module ParetoTests

using Test

import MscThesis.Pareto

@testset "Pareto Domain Tests" begin
    @testset "Constructors Tests" begin
        test_emptyness(pd) =  isempty(pd.dominated_variables) &&
            isempty(pd.nondominated_variables) &&
            isempty(pd.dominated_objectives) &&
            isempty(pd.nondominated_objectives)

        # Successful
        # Empty Initialization
        @test begin
            p = Pareto.Pareto.ParetoResult(1, 1);
            test_emptyness(p) &&
            size(p.nondominated_variables) == size(p.dominated_variables) &&
            size(p.nondominated_objectives) == size(p.dominated_objectives)
        end
        @test begin
            p = Pareto.ParetoResult(2, 2);
            test_emptyness(p) &&
            size(p.nondominated_variables) == size(p.dominated_variables) &&
            size(p.nondominated_objectives) == size(p.dominated_objectives)
        end
        # Default Initialization: Using Vectors
        @test begin
            p = Pareto.ParetoResult([1, 2, 3], [2], [3, 4, 5], [6]);
            p.dominated_variables == reshape([1; 2; 3], (3,1)) &&
            p.dominated_objectives == reshape([2], (1,1)) &&
            p.nondominated_variables == reshape([3; 4; 5], (3,1)) &&
            p.nondominated_objectives == reshape([6], (1,1))
        end
        @test begin
            p = Pareto.ParetoResult([1.0, 2.5, 3.3], [2], [3, 4, 5], [6.9]);
            p.dominated_variables == reshape([1.0; 2.5; 3.3], (3,1)) &&
            p.dominated_objectives == reshape([2], (1,1)) &&
            p.nondominated_variables == reshape([3; 4; 5], (3,1)) &&
            p.nondominated_objectives == reshape([6.9], (1,1))
        end

        # Default Initialization: Using Matrix
        @test begin
            p = Pareto.ParetoResult([1 2 3; 3 4 5], hcat(2, 2, 2), [3 4 5; 6 7 8], hcat(6, 6, 6));
            p.dominated_variables == [1 2 3; 3 4 5] &&
            p.dominated_objectives == hcat(2, 2, 2) &&
            p.nondominated_variables == [3 4 5; 6 7 8] &&
            p.nondominated_objectives == hcat(6, 6, 6)
        end
        @test begin
            p = Pareto.ParetoResult([1, 2, 3], [2, 3], [3, 4, 5], [6, 4]);
            p.dominated_variables == reshape([1 2 3;], (3,1)) &&
            p.dominated_objectives == reshape([2, 3], (2,1)) &&
            p.nondominated_variables == reshape([3 4 5;], (3,1)) &&
            p.nondominated_objectives == reshape([6, 4], (2,1))
        end
        @test begin
            p = Pareto.ParetoResult([1 2 3; 3 4 5; 4 5 2], hcat(2, 3, 3), [3 4 5; 1 2 3; 1 1 1], hcat(6, 4, 3));
            p.dominated_variables == [1 2 3; 3 4 5; 4 5 2] &&
            p.dominated_objectives == hcat(2, 3, 3) &&
            p.nondominated_variables == [3 4 5; 1 2 3; 1 1 1] &&
            p.nondominated_objectives == hcat(6, 4, 3)
        end

        # Method Errors
        @test_throws MethodError Pareto.ParetoResult(1.0, 2.0)
        @test_throws MethodError Pareto.ParetoResult(1.0, 2)
        @test_throws MethodError Pareto.ParetoResult(1, 2.0)
        @test_throws MethodError Pareto.ParetoResult([1], [2.0])

        # Domain Errors
        @test_throws DomainError Pareto.ParetoResult(-1, -2)
        @test_throws DomainError Pareto.ParetoResult(-1, 2)
        @test_throws DomainError Pareto.ParetoResult(1, -1)
        @test_throws DomainError Pareto.ParetoResult(1, 0)
        @test_throws DomainError Pareto.ParetoResult(0, 1)
        @test_throws DomainError Pareto.ParetoResult(0, 0)
        @test_throws DomainError Pareto.ParetoResult([1, 2, 3], [], [3, 4, 3], []);

        # Dimension Mismatch
        @test_throws DimensionMismatch Pareto.ParetoResult([1, 2, 3], [2], [3, 4], [6]);
        @test_throws DimensionMismatch Pareto.ParetoResult([1, 2, 3], [2, 3], [3, 4, 3], [6]);
        @test_throws DimensionMismatch Pareto.ParetoResult([1 2; 4 5], [2, 3], [3 4 3; 4 5 6], [6, 3]);
        @test_throws DimensionMismatch Pareto.ParetoResult([3 4 3; 4 5 6], [2 3 4; 4 5 4], [3 4 3; 4 5 6], [6 3; 1 2]);
    end

    @testset "Selectors Tests" begin
        p1 = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);

        @test Pareto.dominated_variables(p1) == [1 2; 1 1]
        @test Pareto.dominated_objectives(p1) == [2 3; 3 3]

        @test Pareto.nondominated_variables(p1) == [2 2 2; 3 4 5]
        @test Pareto.nondominated_objectives(p1) == [1 3 4; 3 2 1]

        @test Pareto.total_dominated(p1) == 2
        @test Pareto.total_nondominated(p1) == 3

        @test Pareto.variables(p1) == [1 2 2 2 2; 1 1 3 4 5]
        @test Pareto.objectives(p1) == [2 3 1 3 4; 3 3 3 2 1]

        @test Pareto.ParetoFront(p1) == ([2 2 2; 3 4 5], [1 3 4; 3 2 1])
    end
end

@testset "Pareto Dominance Relations Tests" begin
    V0 = [1 3 4; 3 2 1]
    V1 = [0.75 3 4; 0.49 4 5]
    v0, v1, v2 = [4, 4], [0.5, 1], [1, 0.5]
    @testset "Weakly dominates Tests" begin
        @test !Pareto.weakly_dominates(v0, v1)
        @test Pareto.weakly_dominates(v1, v0)
        @test Pareto.weakly_dominates(v2, v0)
        @test !Pareto.weakly_dominates(v0, v2)
        @test !Pareto.weakly_dominates(v2, v1) && !Pareto.weakly_dominates(v1, v2)
        @test !Pareto.weakly_dominates(v0, [4, 3])
        @test !Pareto.weakly_dominates(v0, [3, 4])
        @test Pareto.weakly_dominates([4, 3], v0)
        @test Pareto.weakly_dominates([3, 4], v0)

        @test Pareto.weakly_dominates([1, 1], V0, all)
        @test Pareto.weakly_dominates([1, 1], V0, any)
        @test Pareto.weakly_dominates([2, 1], V0, any)
    end
    @testset "Strongly dominates Tests" begin
        @test Pareto.strongly_dominates(v1, v0)
        @test !Pareto.strongly_dominates(v0, v1)

        @test Pareto.strongly_dominates(v2, v0)
        @test !Pareto.strongly_dominates(v0, v2)

        @test !Pareto.strongly_dominates(v1, v2) && !Pareto.strongly_dominates(v2, v1)
    end
    @testset "Non Dominated Tests" begin
        @test Pareto.is_nondominated(v2, V0)
        @test !Pareto.is_nondominated(v2, V1)

        @test !Pareto.is_nondominated(v0, V0)
        @test !Pareto.is_nondominated(v0, V1)
    end
end

@testset "Remove Tests" begin
    @test begin
        p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
        Pareto.remove_nondominated!(p, 1)
        Pareto.nondominated_variables(p) == [2 2; 4 5] && Pareto.nondominated_objectives(p) == [3 4; 2 1]
    end
    @test begin
        p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
        Pareto.remove_nondominated!(p, [1])
        Pareto.nondominated_variables(p) == [2 2; 4 5] && Pareto.nondominated_objectives(p) == [3 4; 2 1]
    end
    @test begin
        p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
        Pareto.remove_nondominated!(p, [2])
        Pareto.nondominated_variables(p) == [2 2; 3 5] && Pareto.nondominated_objectives(p) == [1 4; 3 1]
    end
    @test begin
        p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
        Pareto.remove_nondominated!(p, [1, 3])
        Pareto.nondominated_variables(p) == reshape([2, 4], (2, 1)) && Pareto.nondominated_objectives(p) == reshape([3, 2], (2,1))
    end
    @test begin
        p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
        Pareto.remove_nondominated!(p, [1, 2, 3])
        isempty(Pareto.nondominated_variables(p)) && isempty(Pareto.nondominated_objectives(p))
    end
end

@testset "Modifiers Tests" begin
    @testset "Push Dominated Tests" begin
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push_dominated!(p, [], [])
            Pareto.dominated_variables(p) == [1 2; 1 1] && Pareto.dominated_objectives(p) == [2 3; 3 3]
        end
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push_dominated!(p, [0, 0], [4, 4])
            Pareto.dominated_variables(p) == [1 2 0; 1 1 0] && Pareto.dominated_objectives(p) == [2 3 4; 3 3 4]
        end
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1; 4 5], [2 3; 3 3], [2 2; 2 3; 4 5], [1 3; 3 2]);
            Pareto.push_dominated!(p, [0, 0, 3], [4, 4])
            Pareto.dominated_variables(p) == [1 2 0; 1 1 0; 4 5 3] &&
            Pareto.dominated_objectives(p) == [2 3 4; 3 3 4] &&
            Pareto.nondominated_variables(p) == [2 2; 2 3; 4 5] &&
            Pareto.nondominated_objectives(p) == [1 3; 3 2]
        end
    end
    @testset "Push NonDominated Tests" begin
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push_nondominated!(p, [], []);
            Pareto.nondominated_variables(p) == [2 2 2; 3 4 5] &&
            Pareto.nondominated_objectives(p) == [1 3 4; 3 2 1]
        end
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push_nondominated!(p, [0, 0], [4, 4])
            Pareto.nondominated_variables(p) == [2 2 2 0; 3 4 5 0] &&
            Pareto.nondominated_objectives(p) == [1 3 4 4; 3 2 1 4]
        end
        @test begin
            p = Pareto.ParetoResult([1 2; 3 1; 1 3], [2 3; 3 3], [2 2 2; 2 2 3; 4 5 3], [1 3 1; 3 2 1]);
            Pareto.push_nondominated!(p, [0, 0, 3], [4, 4]);
            Pareto.nondominated_variables(p) ==  [2 2 2 0; 2 2 3 0; 4 5 3 3] &&
            Pareto.nondominated_objectives(p) == [1 3 1 4; 3 2 1 4] &&
            Pareto.dominated_variables(p) == [1 2; 3 1; 1 3] &&
            Pareto.dominated_objectives(p) == [2 3; 3 3]
        end
    end
    @testset "Push! Tests" begin
        @test_throws DimensionMismatch begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push!(p, [], [])
        end

        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push!(p, [0, 0], [4, 4])
            Pareto.dominated_variables(p) == [1 2 0; 1 1 0] &&
            Pareto.dominated_objectives(p) == [2 3 4; 3 3 4] &&
            Pareto.nondominated_variables(p) == [2 2 2; 3 4 5] &&
            Pareto.nondominated_objectives(p) == [1 3 4; 3 2 1]
        end
        @test begin
            p = Pareto.ParetoResult([1 2; 1 1], [2 3; 3 3], [2 2 2; 3 4 5], [1 3 4; 3 2 1]);
            Pareto.push!(p, [0, 0], [0.5, 0.5])
            Pareto.dominated_variables(p) == [1 2 2 2 2; 1 1 3 4 5] &&
            Pareto.dominated_objectives(p) == [2 3 1 3 4; 3 3 3 2 1] &&
            Pareto.nondominated_variables(p) == reshape([0, 0], (2,1)) &&
            Pareto.nondominated_objectives(p) == reshape([0.5 0.5], (2,1))
        end
    end
end
end #module
