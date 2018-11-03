module OptimizationTests

using Test

import MscThesis

@testset "Variables Tests" begin

    @testset "IntVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(MscThesis.IntVariable(0, 1)) <: MscThesis.AbstractVariable
            @test typeof(MscThesis.IntVariable(0, 50)) <: MscThesis.AbstractVariable
            @test begin i = MscThesis.IntVariable(-25, 25); i.lower_bound == -25 && i.upper_bound == 25 && i.initial_value == 0 end
            @test begin i = MscThesis.IntVariable(-25, 10, 2); i.lower_bound == -25 && i.upper_bound == 10 && i.initial_value == 2 end

            # Method Errors
            @test_throws MethodError MscThesis.IntVariable()
            @test_throws MethodError MscThesis.IntVariable(0)
            @test_throws MethodError MscThesis.IntVariable(0.0, 1)
            @test_throws MethodError MscThesis.IntVariable(0, 1.0)
            @test_throws MethodError MscThesis.IntVariable(0, 1, 0.5)
            @test_throws MethodError MscThesis.IntVariable(0.0, 1.0, 0.5)
            @test_throws MethodError MscThesis.IntVariable([0], [2], [1])

            # Domain Errors
            @test_throws DomainError MscThesis.IntVariable(1, 0)
            @test_throws DomainError MscThesis.IntVariable(-1, -3, -2)
            @test_throws DomainError MscThesis.IntVariable(0, 1, 2)
        end

        @testset "Selectors Tests" begin
            i = MscThesis.IntVariable(0, 20, 10)
            @test MscThesis.lower_bound(i) == 0
            @test MscThesis.upper_bound(i) == 20
            @test MscThesis.initial_value(i) == 10
            @test_throws MethodError MscThesis.values(i)

            i1 = MscThesis.IntVariable(2, 5)
            @test MscThesis.lower_bound(i1) == 2
            @test MscThesis.upper_bound(i1) == 5
            @test MscThesis.initial_value(i1) == 3
            @test_throws MethodError MscThesis.values(i1)
        end

        @testset "Predicates Tests" begin
            @test MscThesis.isIntVariable(MscThesis.IntVariable(0, 20, 10))
            @test MscThesis.isIntVariable(MscThesis.IntVariable(-20, 10))
            @test !MscThesis.isIntVariable(MscThesis.RealVariable(0, 1))
            @test !MscThesis.isIntVariable(MscThesis.SetVariable([0]))
            @test !MscThesis.isIntVariable(nothing)
            @test !MscThesis.isIntVariable(Vector{Int}())
        end
    end

    @testset "RealVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(MscThesis.RealVariable(0.0, 1.0)) <: MscThesis.AbstractVariable
            @test typeof(MscThesis.RealVariable(0, 50)) <: MscThesis.AbstractVariable
            @test begin i = MscThesis.RealVariable(-25.0, 25.0); i.lower_bound == -25.0 && i.upper_bound == 25.0 && i.initial_value == 0 end
            @test begin i = MscThesis.RealVariable(-25.0, 10.0, 2.0); i.lower_bound == -25.0 && i.upper_bound == 10.0 && i.initial_value == 2.0 end
            @test MscThesis.RealVariable(0.0, 1.0) != nothing
            @test MscThesis.RealVariable(0.0, 1.0) == MscThesis.RealVariable(0, 1, 0.5)
            @test MscThesis.RealVariable(0, 1, 0.5) == MscThesis.RealVariable(0.0, 1.0, 0.5)
            @test MscThesis.RealVariable(0.0, 1) == MscThesis.RealVariable(0.0, 1.0)

            # Method Errors
            @test_throws MethodError MscThesis.RealVariable()
            @test_throws MethodError MscThesis.RealVariable(0)
            @test_throws MethodError MscThesis.RealVariable(300.0)
            @test_throws MethodError MscThesis.RealVariable([0.0], [2.0], [1.0])

            # Domain Errors
            @test_throws DomainError MscThesis.RealVariable(1.0, 0.0)
            @test_throws DomainError MscThesis.RealVariable(-1.0, -3.0, -2.0)
            @test_throws DomainError MscThesis.RealVariable(0.0, 1.0, 2.0)
        end

        @testset "Selectors Tests" begin
            r = MscThesis.RealVariable(0.0, 20.0, 10.0)
            @test MscThesis.lower_bound(r) == 0.0
            @test MscThesis.upper_bound(r) == 20.0
            @test MscThesis.initial_value(r) == 10.0
            @test_throws MethodError MscThesis.values(r)

            r1 = MscThesis.RealVariable(2.0, 5.0)
            @test MscThesis.lower_bound(r1) == 2.0
            @test MscThesis.upper_bound(r1) == 5.0
            @test MscThesis.initial_value(r1) == 3.5
            @test_throws MethodError MscThesis.values(r1)
        end

        @testset "Predicates Tests" begin
            @test MscThesis.isRealVariable(MscThesis.RealVariable(0.0, 20.1, 10.0))
            @test MscThesis.isRealVariable(MscThesis.RealVariable(-20.0, 10.0))
            @test MscThesis.isRealVariable(MscThesis.RealVariable(-20, 10))
            @test !MscThesis.isRealVariable(MscThesis.IntVariable(0, 1))
            @test !MscThesis.isRealVariable(MscThesis.SetVariable([0]))
            @test !MscThesis.isRealVariable(nothing)
            @test !MscThesis.isRealVariable(Vector{Real}())
        end
    end

    @testset "SetVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(MscThesis.SetVariable(1, 5, 1, [1, 2, 3, 4, 5])) <: MscThesis.AbstractVariable
            @test typeof(MscThesis.SetVariable(1.0, 3.0, 2.0, [1.0, 2.0, 3.0])) <: MscThesis.AbstractVariable

            @test begin s = MscThesis.SetVariable(-25.0, 25.0, 0, [-25.0, 0, 25.0]); s.lower_bound == -25.0 && s.upper_bound == 25.0 && s.initial_value == 0 end
            @test begin s = MscThesis.SetVariable(-25.0, 10.0, 2.0, collect(-25.0:1:10.0)); s.lower_bound == -25.0 && s.upper_bound == 10.0 && s.initial_value == 2.0 end

            @test begin s = MscThesis.SetVariable(2, collect(1:5)); s.lower_bound == 1 && s.upper_bound == 5 && s.initial_value == 2 && s.values == collect(1:5) end
            @test begin s = MscThesis.SetVariable(collect(1:5)); s.lower_bound == 1 && s.upper_bound == 5 && s.initial_value == s.lower_bound && s.values == collect(1:5) end

            @test MscThesis.SetVariable(0.0, 1.0, 0, collect(0:0.5:1)) != nothing
            @test MscThesis.SetVariable(0.0, 1.0, 0, [0.0, 0.5, 1.0]) == MscThesis.SetVariable(0.0, 1.0, 0, collect(0:0.5:1))

            # Method Errors
            @test_throws MethodError MscThesis.SetVariable()
            @test_throws MethodError MscThesis.SetVariable([])
            @test_throws MethodError MscThesis.SetVariable(0)
            @test_throws MethodError MscThesis.SetVariable(30, 1)
            @test_throws MethodError MscThesis.SetVariable(0.0, 2.0, 1.0)
            @test_throws MethodError MscThesis.SetVariable(0.0, 2.0, 1.0, 0.0:2.0)
            @test_throws MethodError MscThesis.SetVariable([0.0], [2.0], [1.0], [0.0, 1.0, 2.0])

            # Domain Errors
            @test_throws DomainError MscThesis.SetVariable(Vector{Real}())
            @test_throws DomainError MscThesis.SetVariable(0, Vector{Real}())
            @test_throws DomainError MscThesis.SetVariable(1, Vector{Real}())
            @test_throws DomainError MscThesis.SetVariable(3, [1])
            @test_throws DomainError MscThesis.SetVariable(3, [1, 2])
            @test_throws DomainError MscThesis.SetVariable(3, [1, 2, 4])
            @test_throws DomainError MscThesis.SetVariable(-1, 2, 2, [1, 2])
            @test_throws DomainError MscThesis.SetVariable( 1, 3, 2, [1, 2])
            @test_throws DomainError MscThesis.SetVariable( 1, 2, 3, [1, 2])
            @test_throws DomainError MscThesis.SetVariable( 1, 2, 1.5, [1, 2])
            @test_throws DomainError MscThesis.SetVariable(0.0, 3.0, 2.0, [1.0, 2.0, 3.0])
        end

        @testset "Selectors Tests" begin
            s = MscThesis.SetVariable(1, 5, 2, collect(1:5))
            @test MscThesis.lower_bound(s) == 1
            @test MscThesis.upper_bound(s) == 5
            @test MscThesis.initial_value(s) == 2
            @test MscThesis.values(s) == [1, 2, 3, 4, 5]

            s1 = MscThesis.SetVariable(collect(1:5))
            @test MscThesis.lower_bound(s1) == 1
            @test MscThesis.upper_bound(s1) == 5
            @test MscThesis.initial_value(s1) == 1
            @test MscThesis.values(s1) == [1, 2, 3, 4, 5]

            @test s != s1 # Different initial value
            @test s == MscThesis.SetVariable(2, collect(1:5))
        end

        @testset "Predicates Tests" begin
            @test MscThesis.isSetVariable(MscThesis.SetVariable(2, collect(1:5)))
            @test MscThesis.isSetVariable(MscThesis.SetVariable(1, 5, 2, collect(1:5)))
            @test !MscThesis.isSetVariable(MscThesis.IntVariable(-20, 10))
            @test !MscThesis.isSetVariable(MscThesis.RealVariable(0, 1))
            @test !MscThesis.isSetVariable(MscThesis.RealVariable(0.0, 1.0))
            @test !MscThesis.isSetVariable(nothing)
            @test !MscThesis.isSetVariable(Vector{Int64}())
            @test !MscThesis.isSetVariable(Vector{Float64}())
        end
    end
end


# # Tests: Objective
# Objective(identity, 1, :MIN)
# Objective(identity, 1, :MAX)
# Objective(identity, 1, :X)
# Objective(identity, 1)
# Objective(identity, :MIN)
#
# o = Objective(identity, 1)
#
# apply(o, 2)
# coefficient(o) # Should be 1
#
# evaluate(o) # MethodError
# evaluate(o, 2)
#
#
# # Test: Constraints
# # Constructors
# Constraint(identity, 1, ==)
# Constraint(identity, 1, !=)
# Constraint(identity, 1, >=)
# Constraint(identity, 1, <=)
# Constraint(identity, 1, >)
# Constraint(identity, 1, <)
# Constraint(identity, 1, >>)
# Constraint(identity, 1)
# Constraint(identity)
# Constraint(identity, <)
#
# # Selectors
# c = Constraint(identity)
# func(c)
# coefficient(c)
# operator(c)
#
# # Application
# apply(c, 3)
# apply(c, -3)
#
# evaluate(c, 3)
# evaluate(c, -3)
# evaluate(c, 0)
#
# evaluate_penalty(c, 0)
# evaluate_penalty(c, 2)
# evaluate_penalty(c, -2)
# evaluate_penalty(c, 10000000.0)
#
#
# c1 = Constraint(x -> x + 2, -2)
# func(c1)
# coefficient(c1)
# operator(c1)
#
#
# # Application
# apply(c1, 3)
# apply(c1, -3)
#
# evaluate(c1, 3)
# evaluate(c1, -3)
# evaluate(c1, 0)
# evaluate(c1, -2)
#
# evaluate_penalty(c1, 0)
# evaluate_penalty(c1, 2)
# evaluate_penalty(c1, -2)
# evaluate_penalty(c1, 100.2)
#
#
# # Test Model
# Model(1,2,3)
# Model(1,2)
# Model(1)
# Model(1,2,-1)
# Model(1,0)
# Model(0,2,3)
# Model(-1,2,3)
# Model(2,-2,3)
#
# Model([IntVariable(0, 3, 3)],[Objective(identity)])
#
# Model([IntVariable(0, 3, 3)],[Objective(identity)])
# Model([ IntVariable(0, 30, 3),
#         RealVariable(0, 3, 3),
#         RealVariable(0, 3, 3),
#         RealVariable(0, 3, 3),
#         RealVariable(0, 3, 3),
#         RealVariable(0, 3, 3)],
#       [Objective(identity), Objective(identity), Objective(identity), Objective(identity)])
# Model([],[])
# Model([IntVariable(0, 3, 3)])
# Model([Objective(identity)])
#
# Model([IntVariable(0, 3, 3)],[Objective(identity)], [Constraint(identity)])
# Model([Objective(identity)], [Constraint(identity)])
# Model([IntVariable(0, 3, 3)],[Constraint(identity)])
# Model([IntVariable(0, 3, 3), Objective(identity)], [Objective(identity)])
#
#
# # Selectors
# m = Model(1,2,3)
#
# nconstraints(m)
# nobjectives(m)
# nvariables(m)
#
# constraints(m)
# objectives(m)
# variables(m)
#
# m1 = Model([IntVariable(0, 3, 3), RealVariable(2, 3, 2.33)],
# [Objective(identity)])
#
# constraints(m1)
# objectives(m1)
# variables(m1)
#
# nconstraints(m1)
# nobjectives(m1)
# nvariables(m1)
#
# isModel(m)
# isModel(Objective(identity))
# isModel(2)
# isModel(nothing)
#
#
# # Test Solution
# Solution([1,2])
#
# Solution(nothing)
# Solution(Vector{Real}())
#
# s = Solution([1,2])
# variables(s)
# constraints(s)
# objectives(s)
# constraint_violation(s)
#
# nvariables(s)
# nobjectives(s)
# nconstraints(s)
#
# isevaluated(s)
# isfeasible(s)
#
# isSolution(s)
# isSolution(nothing)
# isSolution(2)


end # module
