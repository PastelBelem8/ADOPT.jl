using ScikitLearn

# Generate Sample Data
X = 5 * rand(1, 50)
y = hcat([X[i] for i in 1:50], [-X[i] for i in 1:50])
y1 = sin.(X)

y[1:5:end, 1] += 5 * (0.5 .- rand(10))
y[1:5:end, 2] += 5 * (0.5 .- rand(10))
y1[1:5:end] += 5 * (0.5 .- rand(10))

# Multi Objective
linreg = ScikitLearn.Models.LinearRegression(multi_output=true)
fit!(linreg, X', y)
y_lin = predict(linreg, X')

# Single Objective
linreg1 = ScikitLearn.Models.LinearRegression(multi_output=false)
fit!(linreg1, X, y1)
y_lin1 = predict(linreg1, X)

# No multi_output keyword
# Multi Objective
linreg2 = ScikitLearn.Models.LinearRegression()
fit!(linreg2, X', y)
y_lin2 = predict(linreg2, X')

# Single Objective
linreg3 = ScikitLearn.Models.LinearRegression()
fit!(linreg3, X', y1)
y_lin3 = predict(linreg3, X')


using Plots
# Multi Objective
p1 = scatter(X, y, title="Linear Regression", labels=["Data-Obj1", "Data-Obj2"])
plot!(p1, X, hcat(y_lin[1,:], y_lin[2,:]))

# Single Objective
p2 = scatter(X', y1, label=["Data-Sin"])
plot!(p2, X', y_lin1', label="LinReg-Sin")
p = plot(p1, p2, layout=(2, 1))

savefig("LinearRegression-example.png")
