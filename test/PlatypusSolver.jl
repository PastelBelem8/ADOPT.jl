# Test 1
vars = [IntVariable(0, 20), IntVariable(-20, 0)]
objs = [Objective(x -> x[1] + x[2], :MIN)]
model = Model(vars, objs)

# Step 2. Define the Solver
a_type = NSGAII;
a_params = Dict(:population_size => 10);
solver = PlatypusSolver(a_type, max_eval=300, algorithm_params=a_params)

# Step 3. Solve it!
res = solve(solver, model)
convert(Vector{Solution}, res)

# Test 2
# Step 1. Define the Model
vars = [RealVariable(0, 20), RealVariable(20, 55.99)]
objs = [Objective(x -> x[1] ^ x[2], :MIN)]
cnstrs = [Constraint(x-> x[1], <=)]
model = Model(vars, objs, cnstrs)

# Step 2. Define the Solver
a_type = NSGAII;
a_params = Dict(:population_size => 10);
solver = PlatypusSolver(a_type, max_eval=3000, algorithm_params=a_params)

# Step 3. Solve it!
res = solve(solver, model)


#  Test
# m = Model([IntVariable(0, 100, 2), IntVariable(0, 100, 2)],[Objective(x -> x[1] + x[2])])
# p = convert(Platypus.Problem, m)
# a = Platypus.Algorithm(SPEA2, p)
# convert(Vector{Solution}, Platypus.solve(a, max_eval=3))
# var = Platypus.solve(a, 100)

# sols = Platypus.solve(a, 100, unique_objectives=false)
# Platypus.all_results(a)
# Platypus.get_unique(sols)
# Platypus.get_feasible(sols)
# Platypus.get_nondominated(sols)
# x = convert(Solution, sol)
#
# Platypus.get_feasible(sols[1])
#
# variator = Dict(:name => SBX)
#
# algorithm_params = Dict(:population_size => 30)#,  :variator => variator)
# solver = PlatypusSolver(SPEA2, max_eval=90, algorithm_params=algorithm_params)
# # # model = Model([IntVariable(10, 13, 12), IntVariable(-10, 10, 2)],
# # #               [Objective(x -> (x[1] * x[2]) ^ 2), Objective(x -> x[1] - x[2])])
# model = Model([IntVariable(0, 10, 5), IntVariable(-10, 10, 0)],
#               [Objective(x -> x[1] + x[2])])
# sols = solve(solver, model)
# convert(Solution, sols[1])
# convert(Solution, sols[1])
# convert(Vector{Solution}, sols[1])
