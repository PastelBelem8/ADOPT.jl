# Generate Sample Data
X = sort(5 * rand(40, 1), dims=1)
X = reshape(X, (1, 40))
y = sin.(X)[:]

# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))

size(y)

gpr_model = GaussianProcesses.GP(X, y, GaussianProcesses.MeanZero(), GaussianProcesses.SE(0.0, 0.0), -1e8)
gpr_model1 = GaussianProcesses.GP(X, y, GaussianProcesses.MeanZero(), GaussianProcesses.SE(0.0, 0.0), -1.0)
gpr_model2 = GaussianProcesses.GP(X, y, GaussianProcesses.MeanConst(25.0), GaussianProcesses.SE(0.0, 0.0), -1.0)

kernel = Matern(5/2,[0.0],0.0) + SE(0.0,0.0)
gpr_model3 = GaussianProcesses.GP(X, y, GaussianProcesses.MeanZero(), kernel, -2.0)
y_gpr = predict(gpr_model, X')
y_gpr1 = predict(gpr_model1, X')
y_gpr2 = predict(gpr_model2, X')
y_gpr3 = predict(gpr_model3, X')


using Plots
scatter(X', y, title="Gaussian Process Regression", label="Data")
plot!(X', y_gpr, label="GPR model w/ lognoise -1e8")
plot!(X', y_gpr1, label="GPR model w/ lognoise -1.0")
plot!(X', y_gpr2, label="GPR model w/ lognoise -1.0 and MeanConst")
plot!(X', y_gpr3, label="GPR model w/ lognoise -2.0 and Sum Kernel")


savefig("GPR-example.png")
