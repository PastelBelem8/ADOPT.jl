using RDatasets, LIBSVM

# Load Fisher's classic iris data
iris = dataset("datasets", "iris")

# LIBSVM handles multi-class data automatically using a one-against-one strategy
labels = convert(Vector, iris[:Species])

# First dimension of input data is features; second is istnces
istnces = convert(Array, iris[:, 1:4])'

# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = svmtrain(istnces[:, 1:2:end], labels[1:2:end]);

# Test model on the other half of the data.
(predicted_labels, decision_values) = svmpredict(model, istnces[:, 2:2:end])


using Printf
using Statistics
# Compute accuracy
@printf "Accuracy: %.2f%%\n" Statistics.mean((predicted_labels .== labels[2:2:end]))*100


# Example adapted from
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

# Generate Sample Data
X = sort(5 * rand(40, 1), dims=1)
X = reshape(X, (1, 40))
y = sin.(X)[:]

# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))

# Fit regression model
svr_rbf = LIBSVM.svmtrain(X, y, svmtype=LIBSVM.EpsilonSVR, cost=1e3, gamma=0.1, kernel=Kernel.RadialBasis)
svr_lin = LIBSVM.svmtrain(X, y, svmtype=LIBSVM.EpsilonSVR, cost=1e3, kernel=Kernel.Linear)
svr_poly = LIBSVM.svmtrain(X, y,svmtype=LIBSVM.EpsilonSVR, cost=1e3, degree=2, kernel=Kernel.Polynomial)


y_rbf, _ = svmpredict(svr_rbf, X)
y_lin, _ = svmpredict(svr_lin, X)
y_poly, _ = svmpredict(svr_poly, X)


using Plots
y_rbf
scatter(X', y, title="Support Vector Regression", label="Data")
plot!(X', y_rbf, label="RBF Model")
plot!(X', y_lin, label="Linear Model")
plot!(X', y_poly, label="Polynomial Model")
# plot!(X, y_poly)

savefig("SVR-example.png")
