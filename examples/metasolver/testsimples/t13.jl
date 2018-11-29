# ------------------------------------------------------------------------- #
# Example 13 - Three Variables - One Objective                              #
# @date: 29/11/2018                                                         #
# ------------------------------------------------------------------------- #
using Main.MscThesis
using Main.MscThesis.Metamodels
using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
test_id = 13
# ------------------------------------------------------------------------- #
# Available models are:                                                     #
#   - DecicionTree                                                          #
#   - GPE                                                                   #
#   - LinearRegression                                                      #
#   - MLPRegressor                                                          #
#   - RandomForest                                                          #
#   - SVM                                                                   #
# ------------------------------------------------------------------------- #
model = LinearRegression(multi_output=false)

# !! DO NOT modify the code below
# ..............................................................................
# Step 1. Create the MetaModel, i.e., a definition of the problem to be solved.
# ..............................................................................
# Note that, since this is a MetaProblem involving the approximation of objective
# functions by means of surrogate modellings, this will be a special case of a
# problem, and, therefore, called MetaModel.

# Define a single variable with lower bound -10 and upper bound 10
v1 = IntVariable(-10, 10);
v2 = IntVariable(-10, 10);
v3 = IntVariable(-10, 10);
vars = [v1, v2, v3]
nvars = length(vars)

# Define a single objective described by the f(x) = x²
f0(x) = x[1]^2 + x[2]^2 + x[3]^2
o1 = Objective(f0)

# To automate code below
objs = (o1,)
nobjs = sum(map(nobjectives, objs))

# Create a surrogate associating the model previously defined to the objective
# to be modelled
surrogate = Surrogate(model, objectives=objs)

# Create the Meta Problem that is composed by the variable o1 and the surrogate
# representing the objective
meta_problem = MetaModel(vars, [surrogate])

# ..............................................................................
# Step 2. Create the MetaSolver
# ..............................................................................
# Define the parameters to be used in the initialization of the surrogate
sampling_params = Dict{Symbol, Any}(
    :sampling_function => Sampling.randomMC,
    :nsamples => 5,
    :filename => "examples\\metasolver\\testsimples\\resources\\t$(test_id)-sample.csv"
)

# Specify the algorithm that will explore and find best surrogate solutions
algorithm = SPEA2
algorithm_params = Dict(:population_size => 50)

# We want that the algorithm finds good solutions even if some are dominated (nondominated_only=false)
solver = PlatypusSolver(algorithm,
                        algorithm_params=algorithm_params,
                        nondominated_only = false,
                        max_eval=500)

# Finally create the MetaSolver by specifying its exploitation solver, the number
# of vars, the number of objectives, the number of maximum expensive evaluations
# and the parameters to be used for initiating the surrogate
meta_solver = MetaSolver(solver,
                         nvars=nvars,
                         nobjs=nobjs,
                         max_eval=10,
                         sampling_params=sampling_params)


 # Step 3. Solve it!
 sols = solve(meta_solver, meta_problem)
