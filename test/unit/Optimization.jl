import ADOPT
using Test


@testset "Variables Tests" begin
    @testset "IntVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(ADOPT.IntVariable(0, 1)) <: ADOPT.AbstractVariable
            @test typeof(ADOPT.IntVariable(0, 50)) <: ADOPT.AbstractVariable
            @test begin
                i = ADOPT.IntVariable(-25, 25)
                i.lower_bound == -25 && i.upper_bound == 25 && i.initial_value == 0
            end
            @test begin
                i = ADOPT.IntVariable(-25, 10, 2)
                i.lower_bound == -25 && i.upper_bound == 10 && i.initial_value == 2
            end

            # Method Errors
            @test_throws MethodError ADOPT.IntVariable()
            @test_throws MethodError ADOPT.IntVariable(0)
            @test_throws MethodError ADOPT.IntVariable(0.0, 1)
            @test_throws MethodError ADOPT.IntVariable(0, 1.0)
            @test_throws MethodError ADOPT.IntVariable(0, 1, 0.5)
            @test_throws MethodError ADOPT.IntVariable(0.0, 1.0, 0.5)
            @test_throws MethodError ADOPT.IntVariable([0], [2], [1])

            # Domain Errors
            @test_throws DomainError ADOPT.IntVariable(1, 0)
            @test_throws DomainError ADOPT.IntVariable(-1, -3, -2)
            @test_throws DomainError ADOPT.IntVariable(0, 1, 2)
        end

        @testset "Selectors Tests" begin
            i = ADOPT.IntVariable(0, 20, 10)
            @test ADOPT.lower_bound(i) == 0
            @test ADOPT.upper_bound(i) == 20
            @test ADOPT.initial_value(i) == 10
            @test_throws MethodError ADOPT.values(i)

            i1 = ADOPT.IntVariable(2, 5)
            @test ADOPT.lower_bound(i1) == 2
            @test ADOPT.upper_bound(i1) == 5
            @test ADOPT.initial_value(i1) == 3
            @test_throws MethodError ADOPT.values(i1)
        end

        @testset "Predicates Tests" begin
            @test ADOPT.IntVariable(0, 20, 10) isa ADOPT.IntVariable
            @test ADOPT.IntVariable(-20, 10) isa ADOPT.IntVariable
            @test !(ADOPT.RealVariable(0, 1) isa ADOPT.IntVariable)
            @test !(ADOPT.SetVariable([0]) isa ADOPT.IntVariable)
            @test !(nothing isa ADOPT.IntVariable)
            @test !(Vector{Int}() isa ADOPT.IntVariable)
        end

        @testset "Unscaling Tests" begin
            @test ADOPT.unscale(ADOPT.IntVariable(0, 20, 10), 0.5, 0, 1) == 10
            @test ADOPT.unscale(
                ADOPT.IntVariable(0, 20, 10),
                [0, 0.25, 0.5, 0.75, 1],
                0,
                1,
            ) == [0, 5, 10, 15, 20]
            @test ADOPT.unscale(ADOPT.IntVariable(-10, 10), [-10, -5, 10], -10, 10) ==
                  [-10, -5, 10]
            @test ADOPT.unscale(ADOPT.IntVariable(0, 20, 10), 0.33, 0, 1) == 7
        end
    end

    @testset "RealVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(ADOPT.RealVariable(0.0, 1.0)) <: ADOPT.AbstractVariable
            @test typeof(ADOPT.RealVariable(0, 50)) <: ADOPT.AbstractVariable
            @test begin
                i = ADOPT.RealVariable(-25.0, 25.0)
                i.lower_bound == -25.0 && i.upper_bound == 25.0 && i.initial_value == 0
            end
            @test begin
                i = ADOPT.RealVariable(-25.0, 10.0, 2.0)
                i.lower_bound == -25.0 && i.upper_bound == 10.0 && i.initial_value == 2.0
            end
            @test ADOPT.RealVariable(0.0, 1.0) != nothing
            @test ADOPT.RealVariable(0.0, 1.0) == ADOPT.RealVariable(0, 1, 0.5)
            @test ADOPT.RealVariable(0, 1, 0.5) == ADOPT.RealVariable(0.0, 1.0, 0.5)
            @test ADOPT.RealVariable(0.0, 1) == ADOPT.RealVariable(0.0, 1.0)

            # Method Errors
            @test_throws MethodError ADOPT.RealVariable()
            @test_throws MethodError ADOPT.RealVariable(0)
            @test_throws MethodError ADOPT.RealVariable(300.0)
            @test_throws MethodError ADOPT.RealVariable([0.0], [2.0], [1.0])

            # Domain Errors
            @test_throws DomainError ADOPT.RealVariable(1.0, 0.0)
            @test_throws DomainError ADOPT.RealVariable(-1.0, -3.0, -2.0)
            @test_throws DomainError ADOPT.RealVariable(0.0, 1.0, 2.0)
        end

        @testset "Selectors Tests" begin
            r = ADOPT.RealVariable(0.0, 20.0, 10.0)
            @test ADOPT.lower_bound(r) == 0.0
            @test ADOPT.upper_bound(r) == 20.0
            @test ADOPT.initial_value(r) == 10.0
            @test_throws MethodError ADOPT.values(r)

            r1 = ADOPT.RealVariable(2.0, 5.0)
            @test ADOPT.lower_bound(r1) == 2.0
            @test ADOPT.upper_bound(r1) == 5.0
            @test ADOPT.initial_value(r1) == 3.5
            @test_throws MethodError ADOPT.values(r1)
        end

        @testset "Predicates Tests" begin
            @test ADOPT.RealVariable(0.0, 20.1, 10.0) isa ADOPT.RealVariable
            @test ADOPT.RealVariable(-20.0, 10.0) isa ADOPT.RealVariable
            @test ADOPT.RealVariable(-20, 10) isa ADOPT.RealVariable
            @test !(ADOPT.IntVariable(0, 1) isa ADOPT.RealVariable)
            @test !(ADOPT.SetVariable([0]) isa ADOPT.RealVariable)
            @test !(nothing isa ADOPT.RealVariable)
            @test !(Vector{Real}() isa ADOPT.RealVariable)
        end

        @testset "Unscaling Tests" begin
            @test ADOPT.unscale(ADOPT.RealVariable(0, 20.0), 0.5, 0, 1) == 10
            @test ADOPT.unscale(
                ADOPT.RealVariable(0, 20.0),
                [0, 0.25, 0.5, 0.75, 1],
                0,
                1,
            ) == [0, 5, 10, 15, 20]
            @test ADOPT.unscale(ADOPT.RealVariable(-10.5, 10), [-10, -5, 10], -10, 10) ==
                  [-10.5, -5.375, 10.0]
            @test ADOPT.unscale(ADOPT.RealVariable(0, 20, 10), 0.33, 0, 1) == 0.33 * 20
        end
    end

    @testset "SetVariable Tests" begin
        @testset "Constructors Tests" begin
            # Success
            @test typeof(ADOPT.SetVariable(1, 5, 1, [1, 2, 3, 4, 5])) <:
                  ADOPT.AbstractVariable
            @test typeof(ADOPT.SetVariable(1.0, 3.0, 2.0, [1.0, 2.0, 3.0])) <:
                  ADOPT.AbstractVariable

            @test begin
                s = ADOPT.SetVariable(-25.0, 25.0, 0, [-25.0, 0, 25.0])
                s.lower_bound == -25.0 && s.upper_bound == 25.0 && s.initial_value == 0
            end
            @test begin
                s = ADOPT.SetVariable(-25.0, 10.0, 2.0, collect(-25.0:1:10.0))
                s.lower_bound == -25.0 && s.upper_bound == 10.0 && s.initial_value == 2.0
            end

            @test begin
                s = ADOPT.SetVariable(2, collect(1:5))
                s.lower_bound == 1 &&
                    s.upper_bound == 5 &&
                    s.initial_value == 2 &&
                    s.values == collect(1:5)
            end
            @test begin
                s = ADOPT.SetVariable(collect(1:5))
                s.lower_bound == 1 &&
                    s.upper_bound == 5 &&
                    s.initial_value == s.lower_bound &&
                    s.values == collect(1:5)
            end

            @test ADOPT.SetVariable(0.0, 1.0, 0, collect(0:0.5:1)) != nothing
            @test ADOPT.SetVariable(0.0, 1.0, 0, [0.0, 0.5, 1.0]) ==
                  ADOPT.SetVariable(0.0, 1.0, 0, collect(0:0.5:1))

            # Method Errors
            @test_throws MethodError ADOPT.SetVariable()
            @test_throws MethodError ADOPT.SetVariable([])
            @test_throws MethodError ADOPT.SetVariable(0)
            @test_throws MethodError ADOPT.SetVariable(30, 1)
            @test_throws MethodError ADOPT.SetVariable(0.0, 2.0, 1.0)
            @test_throws MethodError ADOPT.SetVariable(0.0, 2.0, 1.0, 0.0:2.0)
            @test_throws MethodError ADOPT.SetVariable([0.0], [2.0], [1.0], [0.0, 1.0, 2.0])

            # Domain Errors
            @test_throws DomainError ADOPT.SetVariable(Vector{Real}())
            @test_throws DomainError ADOPT.SetVariable(0, Vector{Real}())
            @test_throws DomainError ADOPT.SetVariable(1, Vector{Real}())
            @test_throws DomainError ADOPT.SetVariable(3, [1])
            @test_throws DomainError ADOPT.SetVariable(3, [1, 2])
            @test_throws DomainError ADOPT.SetVariable(3, [1, 2, 4])
            @test_throws DomainError ADOPT.SetVariable(-1, 2, 2, [1, 2])
            @test_throws DomainError ADOPT.SetVariable(1, 3, 2, [1, 2])
            @test_throws DomainError ADOPT.SetVariable(1, 2, 3, [1, 2])
            @test_throws DomainError ADOPT.SetVariable(1, 2, 1.5, [1, 2])
            @test_throws DomainError ADOPT.SetVariable(0.0, 3.0, 2.0, [1.0, 2.0, 3.0])
        end

        @testset "Selectors Tests" begin
            s = ADOPT.SetVariable(1, 5, 2, collect(1:5))
            @test ADOPT.lower_bound(s) == 1
            @test ADOPT.upper_bound(s) == 5
            @test ADOPT.initial_value(s) == 2
            @test ADOPT.values(s) == [1, 2, 3, 4, 5]

            s1 = ADOPT.SetVariable(collect(1:5))
            @test ADOPT.lower_bound(s1) == 1
            @test ADOPT.upper_bound(s1) == 5
            @test ADOPT.initial_value(s1) == 1
            @test ADOPT.values(s1) == [1, 2, 3, 4, 5]

            @test s != s1 # Different initial value
            @test s == ADOPT.SetVariable(2, collect(1:5))
        end

        @testset "Predicates Tests" begin
            @test ADOPT.SetVariable(2, collect(1:5)) isa ADOPT.SetVariable
            @test ADOPT.SetVariable(1, 5, 2, collect(1:5)) isa ADOPT.SetVariable
            @test !(ADOPT.IntVariable(-20, 10) isa ADOPT.SetVariable)
            @test !(ADOPT.RealVariable(0, 1) isa ADOPT.SetVariable)
            @test !(ADOPT.RealVariable(0.0, 1.0) isa ADOPT.SetVariable)
            @test !(nothing isa ADOPT.SetVariable)
            @test !(Vector{Int64}() isa ADOPT.SetVariable)
            @test !(Vector{Float64}() isa ADOPT.SetVariable)
        end

        @testset "Unscaling Tests" begin
            @test ADOPT.unscale(ADOPT.SetVariable(collect(1:10)), 0.5, 0, 1) == 5
            @test ADOPT.unscale(
                ADOPT.SetVariable(collect(1:10)),
                [0, 0.25, 0.5, 0.75, 1],
                0,
                1,
            ) == [1, 3, 5, 8, 10]
            @test ADOPT.unscale(ADOPT.SetVariable(collect(1:10)), [-10, -5, 9], -10, 10) ==
                  [1, 3, 10]
            @test ADOPT.unscale(ADOPT.SetVariable(collect(0:10)), 0.33, 0, 1) == 3
        end
    end
end

@testset "Objectives Tests" begin
    @testset "Constructors Tests" begin
        # Success
        @test begin
            o = ADOPT.Objective(identity)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.sense == :MIN
        end
        @test begin
            o = ADOPT.Objective(identity, 0)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 0 &&
                o.sense == :MIN
        end
        @test begin
            o = ADOPT.Objective(identity, -1)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == -1 &&
                o.sense == :MIN
        end
        @test begin
            o = ADOPT.Objective(identity, 0, :MAX)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 0 &&
                o.sense == :MAX
        end
        @test begin
            o = ADOPT.Objective(identity, :MAX)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.sense == :MAX
        end
        @test begin
            o = ADOPT.Objective(x -> x + 1)
            isa(o.func, Function) && o.func(1) == 2 && o.coefficient == 1 && o.sense == :MIN
        end

        # Method Errors
        @test_throws MethodError ADOPT.Objective()
        @test_throws MethodError ADOPT.Objective(0)
        @test_throws MethodError ADOPT.Objective(identity, identity)

        # Domain Errors
        @test_throws DomainError ADOPT.Objective(identity, 0, :MINIMIZE)
        @test_throws DomainError ADOPT.Objective(identity, 0, :MAXIMIZE)
        @test_throws DomainError ADOPT.Objective(identity, :MINIMIZE)
        @test_throws DomainError ADOPT.Objective(identity, :MAXIMIZE)
    end

    @testset "Selectors Tests" begin
        o1 = ADOPT.Objective(identity, 0, :MIN)
        @test ADOPT.coefficient(o1) == 0
        @test ADOPT.func(o1) != nothing
        @test ADOPT.func(o1) == identity
        @test ADOPT.func(o1)(0) == 0
        @test ADOPT.sense(o1) == :MIN
        @test ADOPT.direction(o1) == -1

        o2 = ADOPT.Objective(exp, 1, :MAX)
        @test ADOPT.coefficient(o2) == 1
        @test ADOPT.func(o2) != nothing
        @test ADOPT.func(o2) == exp
        @test ADOPT.func(o2)(0) == 1
        @test ADOPT.sense(o2) == :MAX
        @test ADOPT.direction(o2) == 1

        @test ADOPT.direction([o1, o2]) == [-1, 1]
    end

    @testset "Predicates Tests" begin
        o1 = ADOPT.Objective(identity, 1, :MIN)
        o2 = ADOPT.Objective(exp, 1, :MAX)

        @test o1 isa ADOPT.Objective
        @test o2 isa ADOPT.Objective
        @test !(2 isa ADOPT.Objective)
        @test !(Vector{Real}() isa ADOPT.Objective)
        @test !(nothing isa ADOPT.Objective)
        @test !(ADOPT.IntVariable(0, 1) isa ADOPT.Objective)

        @test ADOPT.isminimization(o1)
        @test !ADOPT.isminimization(o2)

        @test o1 != o2
        @test o1 == ADOPT.Objective(identity, 1, :MIN)
    end

    @testset "Evaluation Tests" begin
        o = ADOPT.Objective(x -> x^2, 3)
        # Success
        @test ADOPT.apply(o, 2) == 4
        @test ADOPT.apply(o, -1) == 1

        @test ADOPT.evaluate(o, 2) == 4 * 3
        @test ADOPT.evaluate(o, -1) == 1 * 3

        # Method Errors
        @test_throws MethodError ADOPT.apply(o, 2, 3)
        @test_throws MethodError ADOPT.apply(o)

        @test_throws MethodError ADOPT.evaluate(o, 2, 3)
        @test_throws MethodError ADOPT.evaluate(o)
    end
end

@testset "SharedObjective Tests" begin
    @testset "Constructors Tests" begin
        # Success
        @test begin
            g = x -> (1, 2)
            o = ADOPT.SharedObjective(g, [1, 4], [:MIN, :MIN])

            o.n == 2 &&
                isa(o.func, Function) &&
                o.func == g &&
                o.coefficients == [1, 4] &&
                o.senses == [:MIN, :MIN]
        end

        @test begin
            g = x -> (1, 2)
            o = ADOPT.SharedObjective(g, 2)

            o.n == 2 &&
                isa(o.func, Function) &&
                o.func == g &&
                o.coefficients == Real[1, 1] &&
                o.senses == [:MIN, :MIN]
        end

        @test begin
            g = x -> (1, 2)
            coeffs = Real[1, 2, 3]
            o = ADOPT.SharedObjective(g, coeffs)

            o.n == length(coeffs) &&
                isa(o.func, Function) &&
                o.func == g &&
                o.coefficients == coeffs &&
                o.senses == [:MIN, :MIN, :MIN]
        end

        @test begin
            senses = [:MIN, :MIN, :MIN]
            o = ADOPT.SharedObjective(identity, senses)

            o.n == length(senses) &&
                isa(o.func, Function) &&
                o.func == identity &&
                o.coefficients == Real[1, 1, 1] &&
                o.senses == senses
        end

        # Method Errors
        @test_throws MethodError ADOPT.SharedObjective()
        @test_throws MethodError ADOPT.SharedObjective(0)
        @test_throws MethodError ADOPT.SharedObjective(identity, identity)
        @test_throws MethodError ADOPT.SharedObjective(identity, 0, :MINIMIZE)
        @test_throws MethodError ADOPT.SharedObjective(identity, 0, :MAXIMIZE)
        @test_throws MethodError ADOPT.SharedObjective(identity, :MINIMIZE)
        @test_throws MethodError ADOPT.SharedObjective(identity, :MAXIMIZE)

        @test_throws DimensionMismatch ADOPT.SharedObjective(identity, [1, 1], [:MIN])
        @test_throws DomainError ADOPT.SharedObjective(identity, Symbol[])
        @test_throws DomainError ADOPT.SharedObjective(identity, 1)
    end

    @testset "Selectors Tests" begin
        g = x -> (1, 2)
        o1 = ADOPT.SharedObjective(g, [1, 4], [:MIN, :MAX])
        @test ADOPT.coefficient(o1) == [1, 4]
        @test ADOPT.coefficient(o1, 1) == 1
        @test ADOPT.coefficient(o1, :) == [1, 4]
        @test ADOPT.coefficient(o1, 1:2) == ADOPT.coefficients(o1)

        @test ADOPT.func(o1) != nothing
        @test ADOPT.func(o1) == g
        @test ADOPT.func(o1)(0) == (1, 2)

        @test ADOPT.sense(o1) == [:MIN, :MAX]
        @test ADOPT.sense(o1, 1) == :MIN
        @test ADOPT.sense(o1, :) == [:MIN, :MAX]
        @test ADOPT.sense(o1, 1:2) == ADOPT.senses(o1)

        @test ADOPT.direction(o1) == [-1, 1]
        @test ADOPT.direction(o1, 1) == -1
        @test ADOPT.direction(o1, :) == [-1, 1]
        @test ADOPT.direction(o1, 1:2) == ADOPT.direction(o1)

        o2 = ADOPT.SharedObjective(g, [1.75, 0.25], [:MIN, :MAX])
        @test ADOPT.coefficient(o2) == [1.75, 0.25]
        @test ADOPT.func(o2) != nothing
        @test ADOPT.func(o2) == g
        @test ADOPT.func(o2)(0) == (1, 2)
        @test ADOPT.sense(o2) == [:MIN, :MAX]
        @test ADOPT.direction(o2) == [-1, 1]

        o3 = ADOPT.Objective(identity)
        @test ADOPT.direction([o1, o3, o2]) == [-1, 1, -1, -1, 1]
    end

    @testset "Predicates Tests" begin
        f = x -> (-x, 2 * x)
        o1 = ADOPT.SharedObjective(f, [1, 4], [:MIN, :MAX])
        o2 = ADOPT.Objective(exp, :MIN)

        @test !(o1 isa ADOPT.Objective) && (o2 isa ADOPT.Objective)
        @test (o1 isa ADOPT.SharedObjective) && !(o2 isa ADOPT.SharedObjective)
        @test !(2 isa ADOPT.SharedObjective)
        @test !(Vector{Real}() isa ADOPT.SharedObjective)
        @test !(nothing isa ADOPT.SharedObjective)
        @test !(ADOPT.IntVariable(0, 1) isa ADOPT.SharedObjective)

        @test !ADOPT.isminimization(o1)
        @test ADOPT.isminimization(o2)
        @test ADOPT.isminimization(ADOPT.SharedObjective(f, [:MIN, :MIN]))
        @test !ADOPT.isminimization(ADOPT.SharedObjective(f, [:MAX, :MAX]))

        # Comparators
        @test o1 == ADOPT.SharedObjective(f, [1, 4], [:MIN, :MAX])
        @test o1 != ADOPT.SharedObjective(f, [1, 4], [:MAX, :MIN])
        @test o1 != ADOPT.SharedObjective(f, [4, 1], [:MIN, :MAX])
    end

    @testset "Evaluation Tests" begin
        o = ADOPT.SharedObjective(x -> (x^2, -(x^2)), [1, 2], [:MIN, :MAX])
        # Success
        @test ADOPT.apply(o, 2) == (4, -4)
        @test ADOPT.apply(o, -1) == (1, -1)
        # @test ADOPT.apply(o, 2, 3) == (4, -4)

        @test ADOPT.evaluate(o, 2) == -4
        @test ADOPT.evaluate(o, -1) == -1

        # Method Errors
        @test_throws MethodError ADOPT.apply(o, 2, 3)
        @test_throws MethodError ADOPT.apply(o)

        @test_throws MethodError ADOPT.evaluate(o, 2, 3)
        @test_throws MethodError ADOPT.evaluate(o)
    end
end

@testset "Constraints Tests" begin
    @testset "Constructors Tests" begin
        # Success
        @test begin
            o = ADOPT.Constraint(identity)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == (==)
        end
        @test begin
            o = ADOPT.Constraint(identity, 0)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 0 &&
                o.operator == (==)
        end
        @test begin
            o = ADOPT.Constraint(identity, -1)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == -1 &&
                o.operator == (==)
        end
        @test begin
            o = ADOPT.Constraint(identity, ≤)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == ≤
        end
        @test begin
            o = ADOPT.Constraint(identity, <)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == <
        end
        @test begin
            o = ADOPT.Constraint(identity, ≥)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == ≥
        end
        @test begin
            o = ADOPT.Constraint(identity, >)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == >
        end
        @test begin
            o = ADOPT.Constraint(identity, !=)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == !=
        end
        @test begin
            o = ADOPT.Constraint(identity, ==)
            isa(o.func, Function) &&
                o.func == identity &&
                o.coefficient == 1 &&
                o.operator == (==)
        end
        @test begin
            o = ADOPT.Constraint(x -> x + 1)
            isa(o.func, Function) && o.coefficient == 1 && o.operator == (==)
        end

        # Method Errors
        @test_throws MethodError ADOPT.Constraint()
        @test_throws MethodError ADOPT.Constraint(0)
        @test_throws MethodError ADOPT.Constraint(0, identity, ==)
        @test_throws MethodError ADOPT.Constraint(identity, nothing)
        @test_throws MethodError ADOPT.Constraint(identity, nothing, nothing)

        # Domain Errors
        @test_throws DomainError ADOPT.Constraint(identity, 0, isodd)
        @test_throws DomainError ADOPT.Constraint(identity, 0, x -> (x + 1) == 2)
    end

    @testset "Selectors Tests" begin
        o1 = ADOPT.Constraint(identity, 1, >=)
        @test ADOPT.coefficient(o1) == 1
        @test ADOPT.func(o1) != nothing
        @test ADOPT.func(o1) == identity
        @test ADOPT.func(o1)(0) == 0
        @test ADOPT.operator(o1) == >=
        @test ADOPT.operator(o1)(2, 1)
    end

    @testset "Predicates Tests" begin
        c1 = ADOPT.Constraint(identity, 1, ==)
        c2 = ADOPT.Constraint(exp, 1, >=)

        @test c1 isa ADOPT.Constraint
        @test c2 isa ADOPT.Constraint
        @test !(2 isa ADOPT.Constraint)
        @test !(Vector{Real}() isa ADOPT.Constraint)
        @test !(nothing isa ADOPT.Constraint)
        @test !(ADOPT.IntVariable(0, 1) isa ADOPT.Constraint)
        @test !(ADOPT.Objective(identity) isa ADOPT.Constraint)

        @test c1 != c2
        @test c1 == ADOPT.Constraint(identity, 1, ==)
    end

    @testset "Evaluation Tests" begin
        TEST_TOL = 1e-8
        ADOPT.with(ADOPT.ϵ, TEST_TOL) do
            c1 = ADOPT.Constraint(x -> x^2, 3)
            @test ADOPT.apply(c1, 2) == 4
            @test ADOPT.apply(c1, -1) == 1

            @test_throws MethodError ADOPT.apply(c1, 2, 3)
            @test_throws MethodError ADOPT.apply(c1)

            # Test 1: Equality
            @test ADOPT.issatisfied(c1, 0)
            @test !ADOPT.issatisfied(c1, 2)
            @test !ADOPT.issatisfied(c1, -1)

            # Test 1: Greater or equal than
            c2 = ADOPT.Constraint(identity, 2, >=)
            @test ADOPT.issatisfied(c2, 2.0)
            @test !ADOPT.issatisfied(c2, -1)
            @test ADOPT.issatisfied(c2, 0)
            @test ADOPT.issatisfied(c2, -0)

            @test_throws MethodError ADOPT.issatisfied(c1, 2, 3)
            @test_throws MethodError ADOPT.issatisfied(c1)

            # Penalty Constraint
            @test ADOPT.evaluate(c1, 2) - (2 * 3) <= TEST_TOL * 3
            @test ADOPT.evaluate(c1, -2) - (2 * 3) <= TEST_TOL * 3

            @test ADOPT.evaluate(c2, 2) - 0 <= TEST_TOL
            @test ADOPT.evaluate(c2, -1) - (1 * 2) <= TEST_TOL * 2
            @test ADOPT.evaluate(c2, -2) - (2 * 2) <= TEST_TOL * 2
            @test ADOPT.evaluate(c2, 2) != ADOPT.evaluate(c2, -2)
            #^NOTE: Since the penalty is evaluated in terms of magnitude
            # the previous case is considered to be true :)
            @test ADOPT.evaluate(c2, 2) != ADOPT.evaluate(c2, -3)

            @test begin
                c3 = ADOPT.Constraint(identity, 2, ==)
                ADOPT.evaluate(c3, 2) == ADOPT.evaluate(c3, -2)
            end

            @test begin
                c4 = ADOPT.Constraint(identity, 2, !=)
                ADOPT.evaluate(c4, 2) - 4 <= TEST_TOL
            end

            @test begin
                constraints_ = [
                    ADOPT.Constraint(identity, !=),
                    ADOPT.Constraint(identity, ==),
                    ADOPT.Constraint(identity, >=),
                    ADOPT.Constraint(identity, <=),
                    ADOPT.Constraint(identity, >),
                    ADOPT.Constraint(identity, <)
                ]
                values = [1e-2, 2, 3, 5, -1, -3]
                expected_penalty = (2 + 5 + 1) + (3 * TEST_TOL) # VIOLATED Constraints
                ADOPT.evaluate(constraints_, values) - expected_penalty <= TEST_TOL
            end

            @test begin
                constraints_ = [
                    ADOPT.Constraint(identity, !=),
                    ADOPT.Constraint(identity, ==),
                    ADOPT.Constraint(identity, >=),
                    ADOPT.Constraint(identity, <=),
                    ADOPT.Constraint(identity, >),
                    ADOPT.Constraint(identity, <),
                ]
                values = [0.1, 0, 0, 0, 1, -3]
                ADOPT.evaluate(constraints_, values) == 0
            end

        end
    end
end







@testset "Model Tests" begin
    @testset "Constructor Tests" begin
        # Success
        @test begin
            m = ADOPT.Model(1, 1)
            length(m.variables) == 1 &&
                length(m.objectives) == 1 &&
                length(m.constraints) == 0
        end
        @test begin
            m = ADOPT.Model(1, 1, 1)
            length(m.variables) == 1 &&
                length(m.objectives) == 1 &&
                length(m.constraints) == 1
        end
        @test begin
            m = ADOPT.Model([ADOPT.IntVariable(1, 2)], [ADOPT.Objective(x -> x^2)])

            length(m.variables) == 1 &&
                length(m.objectives) == 1 &&
                length(m.constraints) == 0
        end
        @test begin
            m = ADOPT.Model(
                [ADOPT.IntVariable(1, 2)],
                [ADOPT.Objective(x -> x^2)],
                [ADOPT.Constraint(x -> x + 1)],
            )

            length(m.variables) == 1 &&
                length(m.objectives) == 1 &&
                length(m.constraints) == 1
        end
        @test begin
            intvars = [ADOPT.IntVariable(1, 2) for i = 1:1000]
            realvars = [ADOPT.RealVariable(1, 2) for i = 1:1555]
            vars = vcat(intvars, realvars)
            objs = [ADOPT.Objective(identity, i) for i = 1:300]

            m = ADOPT.Model(vars, objs)
            length(m.variables) == 2555 &&
                length(m.objectives) == 300 &&
                length(m.constraints) == 0
        end
        # Domain Error
        @test_throws DomainError ADOPT.Model(-1, 0, 0)
        @test_throws DomainError ADOPT.Model(0, -1, 0)
        @test_throws DomainError ADOPT.Model(0, 0, 0)
        @test_throws DomainError ADOPT.Model(0, 1, 0)
        @test_throws DomainError ADOPT.Model(1, 0, 0)
        @test_throws DomainError ADOPT.Model(1, 1, -1)

        # Method Error
        @test_throws MethodError ADOPT.Model(1, 1, 1.0)
        @test_throws MethodError ADOPT.Model(1, 1.0, -1)
        @test_throws MethodError ADOPT.Model(1.0, 1.0)
    end
    @testset "Selectors Tests" begin
        vars = [ADOPT.IntVariable(0, 1), ADOPT.IntVariable(0, 1), ADOPT.IntVariable(0, 1)]
        objs = [ADOPT.Objective(x -> x .* 2), ADOPT.Objective(x -> (x .* 3) .- 2)]
        constrs = [ADOPT.Constraint(x -> x - 12)]
        m1, m2 = ADOPT.Model(vars, objs), ADOPT.Model(vars, objs, constrs)

        @test ADOPT.nconstraints(m1) == 0
        @test ADOPT.nconstraints(m2) == length(constrs)
        @test ADOPT.nobjectives(m1) == length(objs)
        @test ADOPT.nobjectives(m2) == length(objs)
        @test ADOPT.nvariables(m1) == length(vars)
        @test ADOPT.nvariables(m2) == length(vars)

        @test isempty(ADOPT.constraints(m1))
        @test !isempty(ADOPT.constraints(m2)) && ADOPT.constraints(m2)[1] == constrs[1]

        @test !isempty(ADOPT.objectives(m1)) && ADOPT.objectives(m1) == objs
        @test !isempty(ADOPT.objectives(m2)) && ADOPT.objectives(m2) == objs

        @test !isempty(ADOPT.variables(m1)) && ADOPT.variables(m1) == vars
        @test !isempty(ADOPT.variables(m2)) && ADOPT.variables(m2) == vars

        # Model Default
        m3 = ADOPT.Model(2, 2, 2)

        @test ADOPT.nconstraints(m3) == 2
        @test ADOPT.nobjectives(m3) == 2
        @test ADOPT.nvariables(m3) == 2

        @test begin
            c3 = ADOPT.constraints(m3)
            isa(c3, Vector{ADOPT.Constraint}) &&
                all([!isdefined(c3, i) for i in length(c3)])
        end
        @test begin
            v3 = ADOPT.variables(m3)
            isa(v3, Vector{ADOPT.AbstractVariable}) &&
                all([!isdefined(v3, i) for i in length(v3)])
        end
        @test begin
            o3 = ADOPT.objectives(m3)
            isa(o3, Vector{ADOPT.AbstractObjective}) &&
                all([!isdefined(o3, i) for i in length(o3)])
        end
    end
    @testset "Predicates Tests" begin
        vars1 = [ADOPT.IntVariable(0, 1), ADOPT.RealVariable(0, 1), ADOPT.IntVariable(0, 1)]
        vars2 = [ADOPT.IntVariable(0, 1), ADOPT.IntVariable(0, 1), ADOPT.IntVariable(0, 1)]
        objs = [ADOPT.Objective(x -> x .* 2), ADOPT.Objective(x -> (x .* 3) .- 2)]
        constrs = [ADOPT.Constraint(x -> x - 12)]
        m1, m2, m3 = ADOPT.Model(vars2, objs),
        ADOPT.Model(vars2, objs, constrs),
        ADOPT.Model(2, 2, 2)

        @test ADOPT.isModel(m1)
        @test ADOPT.isModel(m2)
        @test ADOPT.isModel(m3)

        @test !ADOPT.isModel(2)
        @test !ADOPT.isModel(Vector{Real}())
        @test !ADOPT.isModel(nothing)
        @test !ADOPT.isModel(ADOPT.IntVariable(0, 1))
        @test !ADOPT.isModel(ADOPT.Objective(identity))

        @test ADOPT.ismixedtype(ADOPT.Model(vars1, objs))
        @test !ADOPT.ismixedtype(ADOPT.Model(vars2, objs))
        @test !ADOPT.ismixedtype(ADOPT.Model(
            [ADOPT.RealVariable(0, 1), ADOPT.RealVariable(0, 1)],
            objs,
        ))
    end
end

# FIXME - FIX THE TESTS TO REFLECT THE CHANGE OF CONSTRAINTS DATA TYPE TO REAL INSTEAD OF BOOLEAN
@testset "Solution Tests" begin
    @testset "Constructor Tests" begin
        # Success
        @test begin
            s = ADOPT.Solution([1, 1, 1.0])
            length(s.variables) == 3 &&
                length(s.objectives) == 0 &&
                length(s.constraints) == 0 &&
                s.constraint_violation == 0 &&
                s.feasible &&
                !s.evaluated
        end
        @test begin
            s = ADOPT.Solution([1, 1, 1.0], [1.0, 0.3], 3, false)
            length(s.variables) == 3 &&
                length(s.objectives) == 0 &&
                length(s.constraints) == 2 &&
                s.constraint_violation == 3 &&
                !s.feasible &&
                !s.evaluated
        end
        @test begin
            s = ADOPT.Solution([1, 1, 1.0], [1, 2, 3], [1.0, 0.3], 3, false, true)
            length(s.variables) == 3 &&
                length(s.objectives) == 3 &&
                length(s.constraints) == 2 &&
                s.constraint_violation == 3 &&
                !s.feasible &&
                s.evaluated
        end

        # Domain Error
        @test_throws DomainError ADOPT.Solution(Vector{Real}())

        # Method Error
        @test_throws MethodError ADOPT.Solution(nothing)
        @test_throws MethodError ADOPT.Solution([])
        @test_throws MethodError ADOPT.Solution([ADOPT.IntVariable(0, 1)])
        @test_throws MethodError ADOPT.Solution([1, 2, 3], [], 3, false)
    end

    @testset "Selectors Tests" begin
        # Test 1
        s1 = ADOPT.Solution([1, 2, 3.5])

        @test ADOPT.variables(s1) == [1, 2, 3.5]
        @test isempty(ADOPT.objectives(s1))
        @test isempty(ADOPT.constraints(s1))
        @test ADOPT.constraint_violation(s1) == 0

        @test ADOPT.nvariables(s1) == 3
        @test ADOPT.nobjectives(s1) == 0
        @test ADOPT.nconstraints(s1) == 0

        # Test 2
        s2 = ADOPT.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0)

        @test ADOPT.variables(s2) == [1, 2, 3, 4, 5]
        @test ADOPT.objectives(s2) == [29.32, 41.21]
        @test ADOPT.constraints(s2) == [1.0, 0.3]
        @test ADOPT.constraint_violation(s2) == 0

        @test ADOPT.nvariables(s2) == 5
        @test ADOPT.nobjectives(s2) == 2
        @test ADOPT.nconstraints(s2) == 2
    end

    @testset "Predicates Tests" begin
        s1 = ADOPT.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0)
        @test ADOPT.isfeasible(s1)
        @test ADOPT.isevaluated(s1)

        s2 = ADOPT.Solution([1, 2, 3, 4, 5], [1.0, 0.3], 0, true)
        @test ADOPT.isfeasible(s2)
        @test !ADOPT.isevaluated(s2)

        s3 = ADOPT.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0, false, false)
        @test !ADOPT.isfeasible(s3)
        @test !ADOPT.isevaluated(s3)

        s4 = ADOPT.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0, false, true)
        @test !ADOPT.isfeasible(s4)
        @test ADOPT.isevaluated(s4)

        # IsSolution
        @test ADOPT.isSolution(s1)
        @test ADOPT.isSolution(ADOPT.Solution([1, 2, 3, 4, 5]))

        @test !ADOPT.isSolution(2)
        @test !ADOPT.isSolution(Vector{Real}())
        @test !ADOPT.isSolution(nothing)
        @test !ADOPT.isSolution(ADOPT.IntVariable(0, 1))
        @test !ADOPT.isSolution(ADOPT.Objective(identity))
    end
end
