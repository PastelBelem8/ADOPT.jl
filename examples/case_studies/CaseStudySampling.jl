using Main.MscThesis
# Binh and Korn function (Constrained)
binhkorn_vars = [RealVariable(0, 5), RealVariable(0, 3)]

binhkorn_f1(x) = 4x[1]^2 + 4x[2]^2
binhkorn_f2(x) = (x[1] - 5)^2 + (x[2] - 5)^2
binhkorn_objs = [Objective(binhkorn_f1), Objective(binhkorn_f2)]

binhkorn_c1(x) = (x[1] - 5)^2 + x[2]^2 - 25 # <= 0
binhkorn_c2(x) = (x[1] - 8)^2 + (x[2] - 3)^2 - 7.7 # >= 0
binhkorn_cnstrs = [Constraint(binhkorn_c1, <=), Constraint(binhkorn_c2, >=)]

binhkorn = Model(binhkorn_vars, binhkorn_objs, binhkorn_cnstrs)

using Main.MscThesis.Sampling
# Sampling Solver
# Normal sampler
sampling_params = Dict(:sampling_function => randomMC, :nsamples => 30)
simple_solver = SamplingSolver(;algorithm_params=sampling_params,
                        max_eval=200,
                        nondominated_only=true)

binhkorn_sols = solve(simple_solver, binhkorn)

iterative_sampler = SamplingSolver(;algorithm_params=sampling_params,
    sampling_strategy=:iterative,
    max_eval=200,
    nondominated_only=true)

binhkorn_sols = solve(iterative_sampler, binhkorn)
