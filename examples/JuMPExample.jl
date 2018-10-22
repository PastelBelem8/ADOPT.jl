using JuMP

# 1. Creating a Model ------------------------------------------------
# Models are Julia objects
# All variables and constraints are associated with a Model object
# Provide a solver using the `solver` keyword
# http://www.juliaopt.org/JuMP.jl/0.18/refmodel.html#ref-model
m = Model()

# 2. Defining Variables ----------------------------------------------
# Variables are Julia objects, defined using the macro `@variable`.
#   - First argument must be the model.
#   - Second argument declares the variable name and optionally the lower
#       and upper bounds
# These will introduce variable x in the local scope. They must be valid
# Julia variable names.
# http://www.juliaopt.org/JuMP.jl/0.18/refvariable.html#ref-variable7
# Variables, also known as columns or decision variables, are the results
# of the optimization.
# --------------------------------------------------------------------
@variable(m, x)             # No bounds
@variable(m, x >= lb)       # Lower bound only!! lb <= not valid!
@variable(m, x <= ub)       # Upper bound only!!
@variable(m, lb <= x <= ub) # Lower and upper bounds
@variable(m, x == fixedval) # Fixed to a value lb == ub
@variable(m, x, Int)        # Integer restriction
@variable(m, x, Bin)        # Binary restriction
@variable(m, x[1:M, 1:N] >= 0) # Creates an M by N array of variables
@variable(m, x[i=1:10], start=(i/2)) # Initial values

# Variable Categories
t = [:Bin,:Int]
@variable(m, x[i=1:2], category=t[i])
@variable(m, y, category=:SemiCont)

# Anonymous Variables
x = @variable(m) # Equivalent to @variable(m, x)
x = @variable(m, [i=1:3], lowerbound = i, upperbound = 2i) # Equivalent to @variable(m, i <= x[i=1:3] <= 2i)

# E.g.:
s = ["Green", "Blue"]
@variable(m, x[i=1:10,s] >= i, Int) # Bounds depend on variable indices
@variable(m, x[i=1:10,j=1:10; isodd(i+j)] >= 0) # Placing conditions on the index values
# Note that only one condition can be added, although expressions can be built up by using the usual && and || logical operators

# 3. Objectives and Constraints ---------------------------------------
# Use the @constraint and the @objective macros.

@constraint(m, x[i] - s[i] <= 0) # Other options: == and >=
@constraint(m, sum(x[i] for i=1:numLocation) == 1)
@objective(m, Max, 5x+22y + (x+y)/2) # Or Min


# ---------------------------------------------------------------------

# Simple Example
using JuMP
using Clp

m = Model(solver=ClpSolver())
@variable(m, 0 <= x <= 2)
@variable(m, 0 <= y <= 30)
@variable(m, x[i=1:10,j=1:10; isodd(i+j)] >= 0)

@objective(m, Max, 5x + 3y)
@objective(m, Max, 2x-2y)
@constraint(m, 1x + 5y <= 3.00)

print(m)

status = solve(m)
println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
println("y = ", getvalue(y))

# @macroexpand @variable(m, 0 <= x <= 2)
# @macroexpand @objective(m, Max, 5x + 3y)
# @macroexpand @constraint(m, 1x + 5y <= 3.00)
