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
    cmd = `cmd /C racket $(@__DIR__)examples/CaseStudyIP/PavPreto_190309.rkt $(scale(x[1], 1.5)) $(scale(x[2], 6.5)) $(x[3])`
    output = chomp(Base.read(cmd, String))
    output = split(output, '\n')[end]
    parse(Float64, output)
end

# Width, Length, Material
vars = [IntVariable(0, 25), IntVariable(0, 110), IntVariable(0, 3)]
objs = [Objective((x) -> -1 * daylight_function(x), :MIN),
        Objective(cost_function, :MIN)]

problem = Model(vars, objs)


# ------------------------------------------------------------------
# Optimization Parameters
# ------------------------------------------------------------------
# 3 runs
nruns = 3
maxevals = 100
nparticles = 1 # maxevals / 10

EAs_params = Dict(:population_size => nparticles)
PSOs_params = Dict( :leader_size => nparticles,
                    :swarm_size => nparticles,
                    :max_iterations => nparticles * 2)
OMOPSO_params = Dict( :leader_size => nparticles,
                    :swarm_size => nparticles,
                    :max_iterations => nparticles * 2,
                    :epsilons => [2, 50, 0.5])

# >>> Metaheuristics: NSGAII, SPEA2, SMPSO, OMOPSO
benchmark(nruns=1,
            algorithms=[(NSGAII, EAs_params)], #(SPEA2, EAs_params),
                        # (SMPSO, PSOs_params), (OMOPSO, OMOPSO_params)],
          problem=problem, max_evals=1)

# 8 algorithms (1st runs of surrogate models are ran using the previously obtained samples for metaheuristics)
# Surrogates: GP, MLPRegressor, RandomForests, SVR, BayesianRidge
