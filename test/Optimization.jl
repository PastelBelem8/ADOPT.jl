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
@testset "Model Tests" begin
            @testset "Constructor Tests" begin
                        # Success
                        @test begin m = MscThesis.Model(1, 1)
                                    length(m.variables) == 1 &&
                                    length(m.objectives) == 1 &&
                                    length(m.constraints) == 0
                        end
                        @test begin m = MscThesis.Model(1, 1, 1)
                                    length(m.variables) == 1 &&
                                    length(m.objectives) == 1 &&
                                    length(m.constraints) == 1
                        end
                        @test begin m = MscThesis.Model([MscThesis.IntVariable(1, 2)],
                                                        [MscThesis.Objective(x -> x^2)])

                                    length(m.variables) == 1 &&
                                    length(m.objectives) == 1 &&
                                    length(m.constraints) == 0
                        end
                        @test begin m = MscThesis.Model([MscThesis.IntVariable(1, 2)],
                                                        [MscThesis.Objective(x -> x^2)],
                                                        [MscThesis.Constraint(x -> x+1)])

                                    length(m.variables) == 1 &&
                                    length(m.objectives) == 1 &&
                                    length(m.constraints) == 1
                        end
                        @test begin intvars = [MscThesis.IntVariable(1, 2) for i in 1:1000]
                                    realvars= [MscThesis.RealVariable(1, 2) for i in 1:1555]
                                    vars = vcat(intvars, realvars)
                                    objs = [MscThesis.Objective(identity, i) for i in 1:300]

                                    m = MscThesis.Model(vars, objs)
                                    length(m.variables) == 2555 &&
                                    length(m.objectives) == 300 &&
                                    length(m.constraints) == 0
                        end
                        # Domain Error
                        @test_throws DomainError MscThesis.Model(-1, 0, 0)
                        @test_throws DomainError MscThesis.Model(0, -1, 0)
                        @test_throws DomainError MscThesis.Model(0, 0, 0)
                        @test_throws DomainError MscThesis.Model(0, 1, 0)
                        @test_throws DomainError MscThesis.Model(1, 0, 0)
                        @test_throws DomainError MscThesis.Model(1, 1, -1)

                        # Method Error
                        @test_throws MethodError MscThesis.Model(1, 1, 1.0)
                        @test_throws MethodError MscThesis.Model(1, 1.0, -1)
                        @test_throws MethodError MscThesis.Model(1.0, 1.0)
            end
            @testset "Selectors Tests" begin
                        vars = [MscThesis.IntVariable(0, 1), MscThesis.IntVariable(0, 1), MscThesis.IntVariable(0, 1)]
                        objs = [MscThesis.Objective(x -> x .* 2), MscThesis.Objective(x -> (x .* 3) .- 2)]
                        constrs = [MscThesis.Constraint(x-> x - 12)]
                        m1, m2 = MscThesis.Model(vars, objs), MscThesis.Model(vars, objs, constrs)

                        @test MscThesis.nconstraints(m1) == 0
                        @test MscThesis.nconstraints(m2) == length(constrs)
                        @test MscThesis.nobjectives(m1) == length(objs)
                        @test MscThesis.nobjectives(m2) == length(objs)
                        @test MscThesis.nvariables(m1) == length(vars)
                        @test MscThesis.nvariables(m2) == length(vars)

                        @test isempty(MscThesis.constraints(m1))
                        @test !isempty(MscThesis.constraints(m2)) && MscThesis.constraints(m2)[1] == constrs[1]

                        @test !isempty(MscThesis.objectives(m1)) && MscThesis.objectives(m1) == objs
                        @test !isempty(MscThesis.objectives(m2)) && MscThesis.objectives(m2) == objs

                        @test !isempty(MscThesis.variables(m1)) && MscThesis.variables(m1) == vars
                        @test !isempty(MscThesis.variables(m2)) && MscThesis.variables(m2) == vars

                        # Model Default
                        m3 = MscThesis.Model(2, 2, 2)

                        @test MscThesis.nconstraints(m3) == 2
                        @test MscThesis.nobjectives(m3) == 2
                        @test MscThesis.nvariables(m3) == 2

                        @test begin c3 = MscThesis.constraints(m3);
                                    isa(c3, Vector{MscThesis.Constraint}) &&
                                    all([!isdefined(c3, i) for i in length(c3)])
                        end
                        @test begin v3 = MscThesis.variables(m3);
                                    isa(v3, Vector{MscThesis.AbstractVariable}) &&
                                    all([!isdefined(v3, i) for i in length(v3)])
                        end
                        @test begin o3 = MscThesis.objectives(m3);
                                    isa(o3, Vector{MscThesis.AbstractObjective}) &&
                                    all([!isdefined(o3, i) for i in length(o3)])
                        end
            end
            @testset "Predicates Tests" begin
                        vars1 = [MscThesis.IntVariable(0, 1), MscThesis.RealVariable(0, 1),
                                    MscThesis.IntVariable(0, 1)]
                        vars2 = [MscThesis.IntVariable(0, 1), MscThesis.IntVariable(0, 1),
                                                MscThesis.IntVariable(0, 1)]
                        objs = [MscThesis.Objective(x -> x .* 2),
                                    MscThesis.Objective(x -> (x .* 3) .- 2)]
                        constrs = [MscThesis.Constraint(x-> x - 12)]
                        m1, m2, m3 = MscThesis.Model(vars2, objs), MscThesis.Model(vars2, objs, constrs), MscThesis.Model(2, 2, 2)

                        @test MscThesis.isModel(m1)
                        @test MscThesis.isModel(m2)
                        @test MscThesis.isModel(m3)

                        @test !MscThesis.isModel(2)
                        @test !MscThesis.isModel(Vector{Real}())
                        @test !MscThesis.isModel(nothing)
                        @test !MscThesis.isModel(MscThesis.IntVariable(0, 1))
                        @test !MscThesis.isModel(MscThesis.Objective(identity))

                        @test MscThesis.ismixedtype(MscThesis.Model(vars1, objs))
                        @test !MscThesis.ismixedtype(MscThesis.Model(vars2, objs))
                        @test !MscThesis.ismixedtype(MscThesis.Model([MscThesis.RealVariable(0, 1), MscThesis.RealVariable(0, 1)], objs))
            end
end

# FIXME - FIX THE TESTS TO REFLECT THE CHANGE OF CONSTRAINTS DATA TYPE TO REAL INSTEAD OF BOOLEAN
@testset "Solution Tests" begin
            @testset "Constructor Tests" begin
                        # Success
                        @test begin s = MscThesis.Solution([1, 1, 1.0]);
                                    length(s.variables) == 3 &&
                                    length(s.objectives) == 0 &&
                                    length(s.constraints) == 0 &&
                                    s.constraint_violation == 0 &&
                                    s.feasible &&
                                    !s.evaluated
                        end
                        @test begin s = MscThesis.Solution([1, 1, 1.0], [1.0, 0.3], 3, false);
                                    length(s.variables) == 3 &&
                                    length(s.objectives) == 0 &&
                                    length(s.constraints) == 2 &&
                                    s.constraint_violation == 3 &&
                                    !s.feasible &&
                                    !s.evaluated
                        end
                        @test begin s = MscThesis.Solution([1, 1, 1.0], [1,2,3], [1.0, 0.3], 3, false, true);
                                    length(s.variables) == 3 &&
                                    length(s.objectives) == 3 &&
                                    length(s.constraints) == 2 &&
                                    s.constraint_violation == 3 &&
                                    !s.feasible &&
                                    s.evaluated
                        end

                        # Domain Error
                        @test_throws DomainError MscThesis.Solution(Vector{Real}())

                        # Method Error
                        @test_throws MethodError MscThesis.Solution(nothing)
                        @test_throws MethodError MscThesis.Solution([])
                        @test_throws MethodError MscThesis.Solution([MscThesis.IntVariable(0,1)])
                        @test_throws MethodError MscThesis.Solution([1, 2, 3], [], 3, false)
            end
            @testset "Selectors Tests" begin
                        # Test 1
                        s1 = MscThesis.Solution([1, 2, 3.5])

                        @test MscThesis.variables(s1) == [1, 2, 3.5]
                        @test isempty(MscThesis.objectives(s1))
                        @test isempty(MscThesis.constraints(s1))
                        @test MscThesis.constraint_violation(s1) == 0

                        @test MscThesis.nvariables(s1) == 3
                        @test MscThesis.nobjectives(s1) == 0
                        @test MscThesis.nconstraints(s1) == 0

                        # Test 2
                        s2 = MscThesis.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0)

                        @test MscThesis.variables(s2) == [1,2, 3, 4, 5]
                        @test MscThesis.objectives(s2) == [29.32, 41.21]
                        @test MscThesis.constraints(s2) == [1.0, 0.3]
                        @test MscThesis.constraint_violation(s2) == 0

                        @test MscThesis.nvariables(s2) == 5
                        @test MscThesis.nobjectives(s2) == 2
                        @test MscThesis.nconstraints(s2) == 2
            end
            @testset "Predicates Tests" begin
                        s1 = MscThesis.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0)
                        @test MscThesis.isfeasible(s1)
                        @test MscThesis.isevaluated(s1)

                        s2 = MscThesis.Solution([1, 2, 3, 4, 5], [1.0, 0.3], 0, true)
                        @test MscThesis.isfeasible(s2)
                        @test !MscThesis.isevaluated(s2)

                        s3 = MscThesis.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0, false, false)
                        @test !MscThesis.isfeasible(s3)
                        @test !MscThesis.isevaluated(s3)

                        s4 = MscThesis.Solution([1, 2, 3, 4, 5], [29.32, 41.21], [1.0, 0.3], 0, false, true)
                        @test !MscThesis.isfeasible(s4)
                        @test MscThesis.isevaluated(s4)

                        # IsSolution
                        @test MscThesis.isSolution(s1)
                        @test MscThesis.isSolution(MscThesis.Solution([1, 2, 3, 4, 5]))

                        @test !MscThesis.isSolution(2)
                        @test !MscThesis.isSolution(Vector{Real}())
                        @test !MscThesis.isSolution(nothing)
                        @test !MscThesis.isSolution(MscThesis.IntVariable(0, 1))
                        @test !MscThesis.isSolution(MscThesis.Objective(identity))
            end
end

end # module
