# There is not a multi layer perceptron package in Julia.
using Flux

# # Generate Sample Data
# X = randn(2, 40)
# y = (sin.(X[1,:]) .* sin.(X[2, :]))[:]
#
#
# # Add noise to targets
# y[1:5:end] += 3 * (0.5 .- rand(8))
#
# function transform_data(X, y)
#     # [([1, 2], 1), ([3, 4], 2)]
#     data
#     for i in 1:length(y)
#         push!(data, ([X[1, i], X[2, i]], y[i]))
#     end
#     data
# end
#
# data = transform_data(X, y)
# model = Chain(Dense(2, 5, σ), Dense(5, 1))
#
# loss(x, y) = Flux.mse(model(x), y)
#
# opt = ADAM(params(model))
# Flux.train!(loss, data, opt, cb = () -> println("training"))
#
# y_mlp = Tracker.data(model(X))
# y_mlp = reshape(y_mlp, 40)
#
# using Plots
# scatter(X[1,:], y, title="MLP Regression", label="Data")
# plot!(X[1,:], y_mlp, label="MLP model")

using Flux
X = 5 * rand(1, 40)
y = sin.(X)[:]


# Add noise to targets
y[1:5:end] += 3 * (0.5 .- rand(8))


# function transform_data(X, y)
#    # [([1, 2], 1), ([3, 4], 2)]
#    res = []
#    for i in 1:length(y)
#        push!(res, ([X[i]], y[i]))
#    end
#    res
# end
#
# data = transform_data(X', y)
model = Chain( Dense(1, 100, relu),
               Dense(100, 1),
            )

loss(x, y) = Flux.mse(model(x), y)


opt = ADAM(params(model))
Flux.train!(loss, [(X, y)], opt, cb = Flux.throttle(() -> println("training"), 10))
Flux.@epochs 200 Flux.train!(loss, data, opt, cb = Flux.throttle(() -> println("training"), 10))

X'
y_mlp = Tracker.data(model(X'))
y_mlp = reshape(y_mlp, 4000)

a = params(model)

using Plots
scatter(X, y, title="MLP Regression", label="Data")
plot!(X, y_mlp, linewidth=6, label="MLP model (4 layers ReLU)")



[Tracker.data(a[i]) for i in 1:length(a) if iseven(i)]
[Tracker.data(a[i]) for i in 1:length(a) if isodd(i)]
# Flux Examples
W1 = param(rand(3, 5))
b1 = param(rand(3))
layer1(x) = W1 * x .+ b1

W2 = param(rand(2, 3))
b2 = param(rand(2))
layer2(x) = W2 * x .+ b2
