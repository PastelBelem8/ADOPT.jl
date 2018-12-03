# Because the integration of the dependencies is more complex, we won't use this.
# Instead we will use LIBSVM, which mimcs the Scikit-learn API.
using SVR

# Example adapted from
# https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

# Generate Sample Data
X = sort(5 * rand(40, 1), dims=1)
y = sin.(X)[:]

# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))

# Fit regression model
# svm_type: [C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR], defaults to EPSILON_SVR
# kernel_type: [LINEAR, POLY, RBF, SIGMOND, PRECOMPUTED], defaults to RBF


SVR.train(y, X,  C=1e3, gamma=0.1, kernel_type=SVR.RBF)
