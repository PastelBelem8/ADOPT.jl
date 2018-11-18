using GaussianProcesses

# Generate Sample Data
X = sort(5 * rand(1, 40), dims=1)
y = sin.(X)[:]

# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))



gpr_model = GaussianProcesses.GPE(X, y, GaussianProcesses.MeanZero(), GaussianProcesses.SE(0.0, 0.0), -1e8)
gpe_model = GaussianProcesses.GPE(X, y, GaussianProcesses.MeanZero(), GaussianProcesses.SE(0.0, 0.0), -1e8)
gpr_model1 = GaussianProcesses.GPE(X, y, GaussianProcesses.MeanZero(), GaussianProcesses.SE(0.0, 0.0), -1.0)
gpr_model2 = GaussianProcesses.GPE(X, y, GaussianProcesses.MeanConst(25.0), GaussianProcesses.SE(0.0, 0.0), -1.0)

kernel = Matern(5/2,[0.0],0.0) + SE(0.0,0.0)
gpr_model3 = GaussianProcesses.GPE(X, y, GaussianProcesses.MeanZero(), kernel, -2.0)
μ, σ² = predict(gpr_model, X)
y_gpe = predict(gpe_model, X)
y_gpr1 = predict(gpr_model1, X)
y_gpr2 = predict(gpr_model2, X)
y_gpr3 = predict(gpr_model3, X)

using Plots
scatter(X', y, title="Gaussian Process Regression", label="Data")
plot(gpr_model, label="GPR model w/ lognoise -1e8")
scatter!(X', μ.-y, label="GPR model w/ lognoise -1e8")
plot(gpr_model1, label="GPe model w/ lognoise -1e8")
plot(gpr_model2, label="GPR model w/ lognoise -1.0")
plot(gpr_model3, label="GPR model w/ lognoise -1.0 and MeanConst")
savefig("GPR-example.png")

#  Multi dimensional regression
d, n = 2, 50;         #Dimension and number of observations
x = 2π * rand(d, n);                               #Predictors
y = vec(sin.(x[1,:]).*sin.(x[2,:])) + 0.05*rand(n);  #Responses


mZero = MeanZero()                             # Zero mean function
kern = Matern(5/2,[0.0,0.0],0.0) + SE(0.0,0.0)    # Sum kernel with Matern 5/2 ARD kernel
                                               # with parameters [log(ℓ₁), log(ℓ₂)] = [0,0] and log(σ) = 0
                                               # and Squared Exponential Iso kernel with
                                               # parameters log(ℓ) = 0 and log(σ) = 0

gp = GPE(x,y,mZero,kern,-2.0)          # Fit the GP
optimize!(gp)

μ, σ² = predict(gp, x)
scatter(x', y, title="Gaussian Process Regression", label="Data")
plot!(gp, label="GPE")
