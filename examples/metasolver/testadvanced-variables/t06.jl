# ------------------------------------------------------------------------- #
# Example 06 - 2 Variables                                                  #
# @date: 29/11/2018                                                         #
# ------------------------------------------------------------------------- #
using Main.MscThesis
using Main.MscThesis.Metamodels
using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
test_id = 6
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
v1 = IntVariable(-10, 10)
v2 = IntVariable(-10, 10)
vars = [v1, v2]
nvars = length(vars)

# Define a single objective described by the f(x) = x²
f0(x) = x[1]^2
o1 = Objective(f0)

# To automate code below
objs = (o1, )
nobjs = sum(map(nobjectives, objs))

# Create a surrogate associating the model previously defined to the objective
# to be modelled
try
    surrogate = Surrogate(model, objectives=objs, variables_indices=[1, 1])
catch y
    if isa(y, DomainError)
        @info "Test successfull (throw Domain Error): $y"
    else
        @info "Test errored"
        y
    end
end
