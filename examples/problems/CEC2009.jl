# Here, we define two MOO problems
# (https://en.wikipedia.org/wiki/Test_functions_for_optimization):
# - schaffer1
# - kursawe
# - binhkorn
# -------------------------------------------------------------------------

# Schaffer function nº 1 (Unconstrained)
schaffer1_vars = [RealVariable(-10, 10)]

schaffer1_f1(x) = x[1]^2
schaffer1_f2(x) = (x[1] - 2)^2
schaffer1_objs = [Objective(schaffer1_f1), Objective(schaffer1_f2)]

schaffer1 = Model(schaffer1_vars, schaffer1_objs)

# Kursawe function (Unconstrained)
kursawe_vars = [RealVariable(-5, 5), RealVariable(-5, 5), RealVariable(-5, 5)]
kursawe_f1(x) = sum([-10 * exp(-0.2 * √(x[i]^2 + x[i+1]^2)) for i in 1:2])
kursawe_f2(x) = sum([abs(x[i])^0.8 + 5sin(x[i]^3) for i in 1:3])

kursawe_objs = [Objective(kursawe_f1), Objective(kursawe_f2)]
kursawe = Model(kursawe_vars, kursawe_objs)

# Binh and Korn function (Constrained)
binhkorn_vars = [RealVariable(0, 5), RealVariable(0, 3)]

binhkorn_f1(x) = 4x[1]^2 + 4x[2]^2
binhkorn_f2(x) = (x[1] - 5)^2 + (x[2] - 5)^2
binhkorn_objs = [Objective(binhkorn_f1), Objective(binhkorn_f2)]

binhkorn_c1(x) = (x[1] - 5)^2 + x[2]^2 - 25 # <= 0
binhkorn_c2(x) = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7 # >= 0
binhkorn_cnstrs = [Constraint(binhkorn_c1, <=), Constraint(binhkorn_c2, >=)]

binhkorn = Model(binhkorn_vars, binhkorn_objs, binhkorn_cnstrs)
