using Main.MscThesis
# Schaffer function nº 1 (Unconstrained)
schaffer1_vars = [RealVariable(-10, 10)]

schaffer1_f1(x) = x[1]^2
schaffer1_f2(x) = (x[1] - 2)^2
schaffer1_objs = [Objective(schaffer1_f1), Objective(schaffer1_f2)]

schaffer1 = Model(schaffer1_vars, schaffer1_objs)

using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
# Experiment two Algorithms
benchmark(nruns=3,
          algorithms=[(NSGAII,), (NSGAII, Dict(:population_size => 15)), (randomMC, Dict(:nsamples=> 500))],
          problem=schaffer1,
          max_evals=100)


# Test 2. Use Solvers and Algorithms
using Main.MscThesis.ScikitLearnModels: sk_fit!, sk_predict, GaussianProcessRegressor
surrogate = Surrogate(  GaussianProcessRegressor(),
                        objectives=schaffer1_objs,
                        creation_f=sk_fit!,
                        update_f=sk_fit!,
                        evaluation_f=sk_predict)
meta_params = Dict(:sampling_function => randomMC, :nsamples => 30)
optimiser2 = Main.MscThesis.PlatypusSolver(NSGAII, max_eval=500, algorithm_params=Dict(:population_size => 50), nondominated_only=true)
solver = Main.MscThesis.MetaSolver(optimiser2; surrogates=[surrogate], max_eval=200, sampling_params=meta_params, nondominated_only=true)
benchmark(nruns=1, algorithms=[solver, (NSGAII,), (NSGAII, Dict(:population_size => 15)), (randomMC, Dict(:nsamples=> 500))], problem=schaffer1, max_evals=100)
