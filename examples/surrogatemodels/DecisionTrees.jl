using DecisionTree

# Generate Sample Data
X = sort(5 * rand(40, 1), dims=1)
y = sin.(X)[:]

# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))

# Decision Trees
regr_1 = DecisionTreeRegressor()
regr_2 = DecisionTreeRegressor(pruning_purity_threshold=0.05)
regr_3 = RandomForestRegressor(n_trees=20)

fit!(regr_1, X, y);
fit!(regr_2, X, y);
fit!(regr_3, X, y);

y_reg1 = predict(regr_1, X)
y_reg2 = predict(regr_2, X)
y_reg3 = predict(regr_3, X)



using Plots
scatter(X, y, title="Decision Tree Regression", label="Data")
plot!(X, y_reg1, label="DT model")
plot!(X, y_reg2, label="DT model w/ 0.5 prune threshold")
plot!(X, y_reg3, label="RF model w/ 20 trees", linewidth=2)


savefig("DecisionTree-example.png")
