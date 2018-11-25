module MetaSolverTests

using Test
import MscThesis.MetaSolver
# Tests Sampling Routines
# using DelimitedFiles
# X, y = from_file(vars_cols=[1, 2], objs_cols=3, filename="sMC-sample.csv", has_header=true)
# X
# y

# using Main.MscThesis
# v1 = IntVariable(0,  100)
# v2 = IntVariable(-100, 0)
# vars = [v1, v2]
#
# o1 = Objective(x -> x[1] + x[2], 1, :MIN)
#
# using Main.MscThesis.Sampling
# using Main.MscThesis.Metamodels
# # Define the Surrogates
# s1 = LinearRegression(multi_output=false)
# # sampling_params = Dict{Symbol, Any}(
# #     :sampling_function => Sampling.randomMC,
# #     :nsamples => 30,
# #     :filename => "sMC-sample.csv",
# #     :header => ["Var1", "Var2", "Obj1"],
# #     :dlm => ',')
#
# sampling_params = Dict{Symbol, Any}(
#     :filename => "sMC-sample.csv",
#     :vars_cols => [1, 2],
#     :objs_cols => [3])
#
# surrogate_o1 = Surrogate(s1, objectives=(o1,), creation_params=sampling_params)
#
# # Define the Optimiser Solver
# using Main.MscThesis.Platypus
# a_type = NSGAII;
# a_params = Dict(:population_size => 1);
# solver = Main.MscThesis.PlatypusSolver(a_type, max_eval=100, algorithm_params=a_params)
#
# # Define the Meta Solver
# meta_solver = Main.MscThesis.MetaSolver(solver, 2, 1, 1)
#
# # Define the MetaModel
# meta_model = Main.MscThesis.MetaModel(vars, [surrogate_o1])
#
#
# solver = Main.MscThesis.optimiser(meta_solver)
# unsclrs = Main.MscThesis.unscalers(meta_model)
# evals_left = Main.MscThesis.max_evaluations(meta_solver)
# create(s) = Main.MscThesis.create!(s, unsclrs)
#
# cheaper_model1 = Main.MscThesis.cheap_model(meta_model, dynamic=true)
# cheaper_model2 = Main.MscThesis.cheap_model(meta_model, dynamic=false)
#
# # Models
# create(surrogate_o1)
#
# cheaper_model = Main.MscThesis.cheap_model(meta_model)
# expensiv_model = Main.MscThesis.expensive_model(meta_model)
#
# # objs1 = objectives(cheaper_model1)
# # objs2 = objectives(cheaper_model2)
# # obj11 = objs1[1]
# # obj21 = objs2[1]
# #
# # obj11.func([3, 4])
# # obj21.func([3, 4])
# #
# #
#
# args = Vector([-2, 2])
# m = Main.MscThesis.Model(Main.MscThesis.variables(meta_model), Main.MscThesis.original_objectives(meta_model), Main.MscThesis.constraints(meta_model))
# objs = objectives(m)
# s_objs = [evaluate(o, args) for o in objs]
#
#
# oos = Main.MscThesis.original_objectives(meta_model)
# oo_objs = [evaluate(o, args) for o in oos]
#
#
# s = Main.MscThesis.evaluate(m, args)
# Main.MscThesis.variables(s)
# Main.MscThesis.objectives(s)
#
# # Solve it!
# Main.MscThesis.solve(meta_solver, meta_model)
#

####################################
# TEST 2 - Multiple Runs
#####################################
# using Main.MscThesis
# using Main.MscThesis.Metamodels
# using Main.MscThesis.Platypus
# using Main.MscThesis.Sampling
# vars = [IntVariable(0,  100), IntVariable(-100, 0)]
# o1 = Objective(x -> x[1] + x[2], 1, :MIN)
#
# # Define the Surrogates
# s1 = LinearRegression(multi_output=false)
# sampling_params = Dict{Symbol, Any}(
#     :filename => "sMC-sample.csv",
#     :vars_cols => [1, 2],
#     :objs_cols => [3])
#
# surrogate_o1 = Surrogate(s1, objectives=(o1,), creation_params=sampling_params)
# # Define the Optimiser Solver
# a_type = NSGAII;
# a_params = Dict(:population_size => 30);
# solver = Main.MscThesis.PlatypusSolver(a_type, max_eval=100, algorithm_params=a_params)
# # Define the Meta Solver
# meta_solver = Main.MscThesis.MetaSolver(solver, 2, 1, 30)
# # Define the MetaModel
# meta_model = Main.MscThesis.MetaModel(vars, [surrogate_o1])
# # Solve it!
# Main.MscThesis.solve(meta_solver, meta_model)


# solver = Main.MscThesis.optimiser(meta_solver)
# unsclrs = Main.MscThesis.unscalers(meta_model)
# evals_left = Main.MscThesis.max_evaluations(meta_solver)
# create(s) = Main.MscThesis.create!(s, unsclrs)
# correct(solutions) = surrogate ->
# let err = Main.MscThesis.correct!(surrogate, solutions)
#     @info "[$(now())] Retrained surrogate exhibits $(err)% error."
#     err
# end
# using Dates
# cheaper_model1 = Main.MscThesis.cheap_model(meta_model, dynamic=true)
# expensiv_model = Main.MscThesis.expensive_model(meta_model)
# foreach(create,  Main.MscThesis.surrogates(meta_model, unsafe=true))
#
# # While
# candidate_solutions = Main.MscThesis.solve(solver, cheaper_model1)
# solutions = evaluate(expensiv_model, candidate_solutions)
# push!(meta_solver, solutions)
# foreach(correct(solutions),  Main.MscThesis.surrogates(meta_model, unsafe=true))
# Main.MscThesis.ParetoFront(meta_solver)
# # Models



####################################
# TEST 3 - Multiple Target
#####################################
# using Main.MscThesis
# using Main.MscThesis.Metamodels
# using Main.MscThesis.Platypus
# using Main.MscThesis.Sampling
# vars = [IntVariable(1, 100)]
# o1 = Objective(x -> 1/x[1], 1, :MIN)
# o2 = Objective(x -> x[1], 1, :MIN)
#
# # Define the Surrogates
# s1 = LinearRegression(multi_output=false)
# sampling_params = Dict{Symbol, Any}(
#     :sampling_function => Sampling.randomMC,
#     :nsamples => 20,
#     :filename => "sMC-sample-2objs.csv",
#     :header => ["Var1", "Var2", "Obj1", "Obj2"],
#     :dlm => ',')
#
# surrogate_o1 = Main.MscThesis.Surrogate(s1, objectives=(o1, o2), creation_params=sampling_params)
# # Define the Optimiser Solver
# a_type = NSGAII;
# a_params = Dict(:population_size => 5);
# solver = Main.MscThesis.PlatypusSolver(a_type, max_eval=30, algorithm_params=a_params)
# # Define the Meta Solver
# meta_solver = Main.MscThesis.MetaSolver(solver, 1, 2, 30)
# # Define the MetaModel
# meta_model = Main.MscThesis.MetaModel(vars, [surrogate_o1])
# # Solve it!
# Main.MscThesis.solve(meta_solver, meta_model)
#
# #=
# solver = Main.MscThesis.optimiser(meta_solver)
# unsclrs = Main.MscThesis.unscalers(meta_model)
# evals_left = Main.MscThesis.max_evaluations(meta_solver)
# create(s) = Main.MscThesis.create!(s, unsclrs)
# correct(solutions) = surrogate ->
#                     let err = Main.MscThesis.correct!(surrogate, solutions)
#                         @info "[$(now())] Retrained surrogate exhibits $(err)% error."
#                         err
#                     end
# foreach(create,  Main.MscThesis.surrogates(meta_model, unsafe=true))
# using Dates
# cheaper_model1 = Main.MscThesis.cheap_model(meta_model, dynamic=true)
# args = [39, -99]
# nobjectives(cheaper_model1)
# objs = objectives(cheaper_model1)
# println(Main.MscThesis.objectives(evaluate(cheaper_model1, Solution(apply(objs[1], args)[:]))))
# expensiv_model = Main.MscThesis.expensive_model(meta_model)
# println(Main.MscThesis.objectives(evaluate(expensiv_model, Solution(apply(objs[1], args)[:]))))
#
# # While
# candidate_solutions = Main.MscThesis.solve(solver, cheaper_model1)
# solutions = evaluate(expensiv_model, candidate_solutions)
# push!(meta_solver, solutions)
# foreach(correct(solutions),  Main.MscThesis.surrogates(meta_model, unsafe=true))
# =#
# Main.MscThesis.ParetoFront(meta_solver)
# Main.MscThesis.results(meta_solver)


end # module
