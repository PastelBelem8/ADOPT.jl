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
@testset "Objectives Tests" begin
            @testset "Constructors Tests" begin
                        # Success
                        @test begin o = MscThesis.Objective(identity); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.sense == :MIN end
                        @test begin o = MscThesis.Objective(identity, 0);  isa(o.func, Function) && o.func == identity && o.coefficient == 0 && o.sense == :MIN end
                        @test begin o = MscThesis.Objective(identity, -1);  isa(o.func, Function) && o.func == identity && o.coefficient == -1 && o.sense == :MIN end
                        @test begin o = MscThesis.Objective(identity, 0, :MAX); isa(o.func, Function) && o.func == identity && o.coefficient == 0 && o.sense == :MAX end
                        @test begin o = MscThesis.Objective(identity, :MAX); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.sense == :MAX end
                        @test begin o = MscThesis.Objective(x-> x + 1); isa(o.func, Function) && o.func(1) == 2 && o.coefficient == 1 && o.sense == :MIN end

                        # Method Errors
                        @test_throws MethodError MscThesis.Objective()
                        @test_throws MethodError MscThesis.Objective(0)
                        @test_throws MethodError MscThesis.Objective(identity, identity)

                        # Domain Errors
                        @test_throws DomainError MscThesis.Objective(identity, 0, :MINIMIZE)
                        @test_throws DomainError MscThesis.Objective(identity, 0, :MAXIMIZE)
                        @test_throws DomainError MscThesis.Objective(identity, :MINIMIZE)
                        @test_throws DomainError MscThesis.Objective(identity, :MAXIMIZE)
            end

            @testset "Selectors Tests" begin
                o1 = MscThesis.Objective(identity, 0, :MIN)
                @test MscThesis.coefficient(o1) == 0
                @test MscThesis.func(o1) != nothing
                @test MscThesis.func(o1) == identity
                @test MscThesis.func(o1)(0) == 0
                @test MscThesis.sense(o1) == :MIN
                @test MscThesis.direction(o1) == -1

                o2 = MscThesis.Objective(exp, 1, :MAX)
                @test MscThesis.coefficient(o2) == 1
                @test MscThesis.func(o2) != nothing
                @test MscThesis.func(o2) == exp
                @test MscThesis.func(o2)(0) == 1
                @test MscThesis.sense(o2) == :MAX
                @test MscThesis.direction(o2) == 1

                @test MscThesis.directions([o1, o2]) == [-1, 1]
            end

            @testset "Predicates Tests" begin
                 o1 = MscThesis.Objective(identity, 1, :MIN)
                 o2 = MscThesis.Objective(exp, 1, :MAX)

                 @test MscThesis.isObjective(o1)
                 @test MscThesis.isObjective(o2)
                 @test !MscThesis.isObjective(2)
                 @test !MscThesis.isObjective(Vector{Real}())
                 @test !MscThesis.isObjective(nothing)
                 @test !MscThesis.isObjective(MscThesis.IntVariable(0, 1))

                 @test MscThesis.isminimization(o1)
                 @test !MscThesis.isminimization(o2)

                 @test o1 != o2
                 @test o1 == MscThesis.Objective(identity, 1, :MIN)
            end

            @testset "Evaluation Tests" begin
                o = MscThesis.Objective(x -> x^2, 3)
                # Success
                @test MscThesis.apply(o, 2) == 4
                @test MscThesis.apply(o, -1) == 1

                @test MscThesis.evaluate(o, 2) == 4 * 3
                @test MscThesis.evaluate(o, -1) == 1 * 3

                # Method Errors
                @test_throws MethodError MscThesis.apply(o, 2, 3)
                @test_throws MethodError MscThesis.apply(o)

                @test_throws MethodError MscThesis.evaluate(o, 2, 3)
                @test_throws MethodError MscThesis.evaluate(o)

            end
end
@testset "Constraints Tests" begin
            @testset "Constructors Tests" begin
                        # Success
                        @test begin o = MscThesis.Constraint(identity); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == (==) end
                        @test begin o = MscThesis.Constraint(identity, 0);  isa(o.func, Function) && o.func == identity && o.coefficient == 0 && o.operator == (==) end
                        @test begin o = MscThesis.Constraint(identity, -1);  isa(o.func, Function) && o.func == identity && o.coefficient == -1 && o.operator == (==) end
                        @test begin o = MscThesis.Constraint(identity, ≤); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == ≤ end
                        @test begin o = MscThesis.Constraint(identity, <); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == < end
                        @test begin o = MscThesis.Constraint(identity, ≥); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == ≥ end
                        @test begin o = MscThesis.Constraint(identity, >); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == > end
                        @test begin o = MscThesis.Constraint(identity, !=); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == != end
                        @test begin o = MscThesis.Constraint(identity, ==); isa(o.func, Function) && o.func == identity && o.coefficient == 1 && o.operator == (==) end
                        @test begin o = MscThesis.Constraint(x-> x + 1); isa(o.func, Function) && o.coefficient == 1 && o.operator == (==) end

                        # Method Errors
                        @test_throws MethodError MscThesis.Constraint()
                        @test_throws MethodError MscThesis.Constraint(0)
                        @test_throws MethodError MscThesis.Constraint(0, identity, ==)
                        @test_throws MethodError MscThesis.Constraint(identity, nothing)
                        @test_throws MethodError MscThesis.Constraint(identity, nothing, nothing)

                        # Domain Errors
                        @test_throws DomainError MscThesis.Constraint(identity, 0, isodd)
                        @test_throws DomainError MscThesis.Constraint(identity, 0, x -> (x + 1) == 2)
            end

            @testset "Selectors Tests" begin
                        o1 = MscThesis.Constraint(identity, 1, >=)
                        @test MscThesis.coefficient(o1) == 1
                        @test MscThesis.func(o1) != nothing
                        @test MscThesis.func(o1) == identity
                        @test MscThesis.func(o1)(0) == 0
                        @test MscThesis.operator(o1) == >=
                        @test MscThesis.operator(o1)(2, 1)
            end

            @testset "Predicates Tests" begin
                        c1 = MscThesis.Constraint(identity, 1, ==)
                        c2 = MscThesis.Constraint(exp, 1, >=)

                        @test MscThesis.isConstraint(c1)
                        @test MscThesis.isConstraint(c2)
                        @test !MscThesis.isConstraint(2)
                        @test !MscThesis.isConstraint(Vector{Real}())
                        @test !MscThesis.isConstraint(nothing)
                        @test !MscThesis.isConstraint(MscThesis.IntVariable(0, 1))
                        @test !MscThesis.isConstraint(MscThesis.Objective(identity))

                        @test c1 != c2
                        @test c1 == MscThesis.Constraint(identity, 1, ==)
            end

            @testset "Evaluation Tests" begin
                        # Success
                        c1 = MscThesis.Constraint(x -> x^2, 3)
                        @test MscThesis.apply(c1, 2) == 4
                        @test MscThesis.apply(c1, -1) == 1

                        # Test 1: Equality
                        @test MscThesis.issatisfied(c1, 0)
                        @test !MscThesis.issatisfied(c1, 2)
                        @test !MscThesis.issatisfied(c1, -1)

                        # Test 1: Greater or equal than
                        c2 = MscThesis.Constraint(x -> x, 2, >=)
                        @test MscThesis.issatisfied(c2, 2.0)
                        @test !MscThesis.issatisfied(c2, -1)
                        @test MscThesis.issatisfied(c2, 0)
                        @test MscThesis.issatisfied(c2, -0)

                        # Penalty Constraint
                        @test MscThesis.evaluate_penalty(c1, 2) == 4 * 3
                        @test MscThesis.evaluate_penalty(c1, -2) == 4 * 3

                        @test MscThesis.evaluate_penalty(c2, 2) == 0
                        @test MscThesis.evaluate_penalty(c2, -1) == 2
                        @test MscThesis.evaluate_penalty(c2, 2)  != MscThesis.evaluate_penalty(c2, -2)
                        @test begin
                                    c3 = MscThesis.Constraint(x -> x, 2, ==);
                                    MscThesis.evaluate_penalty(c3, 2) ==
                                    MscThesis.evaluate_penalty(c3, -2)
                        end

                        # Method Errors
                        @test_throws MethodError MscThesis.apply(c1, 2, 3)
                        @test_throws MethodError MscThesis.apply(c1)
                        @test_throws MethodError MscThesis.evaluate_penalty(MscThesis.Constraint(identity, 2, !=), 2)

                        @test_throws MethodError MscThesis.issatisfied(c1, 2, 3)
                        @test_throws MethodError MscThesis.issatisfied(c1)
            end
end

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
