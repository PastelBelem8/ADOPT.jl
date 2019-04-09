using MScThesis

# Define variables
vars = [RealVariable(-10, 10)]

# Define objectives
f1(x) = x[1]^2
f2(x) = (x[1] - 2)^2
objs = [Objective(f1, :MIN), Objective(f2, :MIN)]

# Create the optimization problem
model = Model(vars, objs)

# Optimize it!
solve(NSGAII, model, max_evals=100)
