
# Tests: Objective
Objective(identity, 1, :MIN)
Objective(identity, 1, :MAX)
Objective(identity, 1, :X)
Objective(identity, 1)
Objective(identity, :MIN)

o = Objective(identity, 1)

apply(o, 2)
coefficient(o) # Should be 1

evaluate(o) # MethodError
evaluate(o, 2)


# Test: Constraints
# Constructors
Constraint(identity, 1, ==)
Constraint(identity, 1, !=)
Constraint(identity, 1, >=)
Constraint(identity, 1, <=)
Constraint(identity, 1, >)
Constraint(identity, 1, <)
Constraint(identity, 1, >>)
Constraint(identity, 1)
Constraint(identity)
Constraint(identity, <)

# Selectors
c = Constraint(identity)
func(c)
coefficient(c)
operator(c)

# Application
apply(c, 3)
apply(c, -3)

evaluate(c, 3)
evaluate(c, -3)
evaluate(c, 0)

evaluate_penalty(c, 0)
evaluate_penalty(c, 2)
evaluate_penalty(c, -2)
evaluate_penalty(c, 10000000.0)


c1 = Constraint(x -> x + 2, -2)
func(c1)
coefficient(c1)
operator(c1)


# Application
apply(c1, 3)
apply(c1, -3)

evaluate(c1, 3)
evaluate(c1, -3)
evaluate(c1, 0)
evaluate(c1, -2)

evaluate_penalty(c1, 0)
evaluate_penalty(c1, 2)
evaluate_penalty(c1, -2)
evaluate_penalty(c1, 100.2)


# Test Model
Model(1,2,3)
Model(1,2)
Model(1)
Model(1,2,-1)
Model(1,0)
Model(0,2,3)
Model(-1,2,3)
Model(2,-2,3)

Model([IntVariable(0, 3, 3)],[Objective(identity)])

Model([IntVariable(0, 3, 3)],[Objective(identity)])
Model([ IntVariable(0, 30, 3),
        RealVariable(0, 3, 3),
        RealVariable(0, 3, 3),
        RealVariable(0, 3, 3),
        RealVariable(0, 3, 3),
        RealVariable(0, 3, 3)],
      [Objective(identity), Objective(identity), Objective(identity), Objective(identity)])
Model([],[])
Model([IntVariable(0, 3, 3)])
Model([Objective(identity)])

Model([IntVariable(0, 3, 3)],[Objective(identity)], [Constraint(identity)])
Model([Objective(identity)], [Constraint(identity)])
Model([IntVariable(0, 3, 3)],[Constraint(identity)])
Model([IntVariable(0, 3, 3), Objective(identity)], [Objective(identity)])


# Selectors
m = Model(1,2,3)

nconstraints(m)
nobjectives(m)
nvariables(m)

constraints(m)
objectives(m)
variables(m)

m1 = Model([IntVariable(0, 3, 3), RealVariable(2, 3, 2.33)],
[Objective(identity)])

constraints(m1)
objectives(m1)
variables(m1)

nconstraints(m1)
nobjectives(m1)
nvariables(m1)

isModel(m)
isModel(Objective(identity))
isModel(2)
isModel(nothing)
