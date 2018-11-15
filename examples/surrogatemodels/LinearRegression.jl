using ScikitLearn
# Generate Sample Data
X = sort(5 * rand(50, 1), dims=1)
y = [X[i] for i in 1:50]

# Add noise to targets
y[1:2:end] += 5 * (0.5 .- rand(25))

linreg = ScikitLearn.Models.LinearRegression(multi_output=false)
fit!(linreg, X', y)

y_lin = predict(linreg, X')

using Plots
scatter(X, y, title="Linear Regression", label="Data")
plot!(X, y_lin', label="Linear Regression model")

savefig("LinearRegression-example.png")


# Other Example

using RDatasets: dataset
using ScikitLearn

X = randn(3, 50)
y = rand(0:1, 50)
