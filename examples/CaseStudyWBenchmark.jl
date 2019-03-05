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
benchmark(nruns=1,
          algorithms=[(NSGAII,), (NSGAII, Dict(:population_size => 15)), (randomMC, Dict(:nsamples=> 500))],
          problem=schaffer1,
          max_evals=100)
