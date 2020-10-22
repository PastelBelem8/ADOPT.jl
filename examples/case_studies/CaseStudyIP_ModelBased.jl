using Main.MscThesis
using Main.MscThesis.Sampling
using Main.MscThesis.Platypus
# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
scale(x, min, step=0.1) = min + step*x

cost_function(x) = let
    p1 = scale(x[1], 1.5) * scale(x[2], 6.5) * 185
    p2 = (scale(x[1], 1.5) + scale(x[2], 6.5)) * 3 * 80
    p1 + p2
end

daylight_function(x) = let
    cmd = `cmd /C racket $(@__DIR__)examples/CaseStudyIP/PavPreto_190309.rkt $(scale(x[1], 1.5)) $(scale(x[2], 6.5)) $(round(Int, x[3]))`
    output = chomp(Base.read(cmd, String))
    output = split(output, '\n')[end]
    parse(Float64, output)
end

# Width, Length, Material
vars = [IntVariable(0, 25), IntVariable(0, 110), IntVariable(0, 2)]
objs = [Objective((x) -> -1 * daylight_function(x), :MIN),
        Objective(cost_function, :MIN)]

problem = Model(vars, objs)

# ------------------------------------------------------------------
# Optimization Parameters
# ------------------------------------------------------------------

using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
using Main.MscThesis.ScikitLearnModels
nruns = 3
maxevals = 200
iter = 20
nparticles = div(maxevals, iter)
nevals_mtsolver = nparticles * iter * 2

# Metaheuristic
NSGAII_solver() = let
  params = Dict(:population_size => nparticles)
  Main.MscThesis.PlatypusSolver(NSGAII, max_eval=nevals_mtsolver, algorithm_params=params, nondominated_only=true)
end

SPEA2_solver() = let
  params = Dict(:population_size => nparticles)
  Main.MscThesis.PlatypusSolver(SPEA2, max_eval=nevals_mtsolver, algorithm_params=params, nondominated_only=true)
end

pso_solver() = let
  params = Dict(:leader_size => nparticles,
                :swarm_size => nparticles,
                :mutation_probability => 0.3,
                :mutation_perturbation => 0.5)
  PlatypusSolver(SMPSO, max_eval=nevals_mtsolver, algorithm_params=params, nondominated_only=true)
end

random_solver() = let
  params = Dict(:sampling_function => randomMC,
                :nsamples => 1000)
  SamplingSolver(;algorithm_params=params, max_eval=nevals_mtsolver, nondominated_only=true)
end

# Meta Solver
meta_solver(metamodel, solver) = let
  params = Dict(:sampling_function => randomMC, :nsamples => 75)
  surrogate = Surrogate(  metamodel, objectives=objs, creation_f=sk_fit!,
                          update_f=sk_fit!, evaluation_f=sk_predict)
  MetaSolver(solver; surrogates=[surrogate], max_eval=maxevals, sampling_params=params, nondominated_only=true)
end

# Test 1 - GPR
# gpr_1 = meta_solver(GaussianProcessRegressor(), pso_solver())
gpr_2 = meta_solver(GaussianProcessRegressor(), NSGAII_solver())
gpr_3 = meta_solver(GaussianProcessRegressor(), SPEA2_solver())
gpr_4 = meta_solver(GaussianProcessRegressor(), random_solver())

# Test 2 - Random Forest
#random_forest_1 = meta_solver(RandomForestRegressor(), pso_solver())
random_forest_2 = meta_solver(RandomForestRegressor(), NSGAII_solver())
random_forest_3 = meta_solver(RandomForestRegressor(), SPEA2_solver())
random_forest_4 = meta_solver(RandomForestRegressor(), random_solver())

# Test 3 - SVR
# mlp_1 = (X, y) -> meta_solver(MLPRegressor(), pso_solver())
mlp_2 = meta_solver(MLPRegressor(), NSGAII_solver())
mlp_3 = meta_solver(MLPRegressor(), SPEA2_solver())
mlp_4 = meta_solver(MLPRegressor(), random_solver())

EAs_params = Dict(:population_size => nparticles)
PSOs_params = Dict( :leader_size => nparticles,
                    :swarm_size => nparticles,
                    :max_iterations => nparticles * 2)
OMOPSO_params = Dict( :leader_size => nparticles,
                    :swarm_size => nparticles,
                    :max_iterations => nparticles * 2,
                    :epsilons => [0.5, 5, 0.5])

solvers = [# gpr_2,
          # gpr_3,
          gpr_4,
          # random_forest_2,
          # random_forest_3,
          # random_forest_4,
          ]
Main.MscThesis.benchmark(nruns=1,
                         algorithms=solvers,
                         problem=problem,
                         max_evals=maxevals)
