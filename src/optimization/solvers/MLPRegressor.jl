using Flux
using Dates
using Random

using ScikitLearnBase
import ScikitLearnBase: fit!, predict

# ---------------------------------------------------------------------------
# Flux Utils
# ---------------------------------------------------------------------------
coeffs(m::Chain) = map(Flux.Tracker.data, params(m)[1:2:end])

# Flux Optimisers
ADADelta(;ρ=0.9, ϵ=1e-8, decay=0) =
    (ps) -> Flux.ADADelta(ps, ρ=ρ, ϵ=ϵ, decay=decay)
ADAGrad(η=0.001; ϵ=1e-8, decay=0) =
    (ps) -> Flux.ADAGrad(ps, η, ϵ=ϵ, decay=decay)
ADAM(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8, decay=0) =
    (ps) -> Flux.ADAM(ps, η, β1=β1, β2=β2, ϵ=ϵ, decay=decay)
AdaMax(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8, decay=0) =
    (ps) -> Flux.AdaMax(ps, η, β1=β1, β2=β2, ϵ=ϵ, decay=decay)
ADAMW(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8, decay=0) =
    (ps) -> Flux.ADAMW(ps, η, β1=β1, β2=β2, ϵ=ϵ, decay=decay)
AMSGrad(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8, decay=0) =
    (ps) -> Flux.AMSGrad(ps, η, β1=β1, β2=β2, ϵ=ϵ, decay=decay)
Momentum(η=0.001; ρ=0.9, decay=0) =
    (ps) -> Flux.Momentum(ps, η, ρ=ρ, decay=decay)
NADAM(η=0.001; β1=0.9, β2=0.999, ϵ=1e-8, decay=0) =
    (ps) -> Flux.NADAM(ps, η, β1=β1, β2=β2, ϵ=ϵ, decay=decay)
Nesterov(η=0.001; ρ=0.9, decay=0) =
    (ps) -> Flux.Nesterov(ps, η, ρ=ρ, decay=decay)
RMSProp( η=0.001; ρ=0.9, ϵ=1e-8, decay=0) =
    (ps) -> Flux.RMSProp(ps, η, ρ=ρ, ϵ=ϵ, decay=decay)
SGD(η=0.001; decay=0) =
    (ps) -> Flux.SGD(ps, decay=decay)

export  ADADelta, ADAGrad, ADAM, AdaMax, ADAMW, AMSGrad,
        Momentum, NADAM, Nesterov, RMSProp, SGD

# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------
mutable struct MLPRegressor
    tol::Real
    epochs::Int
    epochs_nochange::Int
    batch_size::Int
    batch_shuffle::Bool

    ninputs::Int
    noutputs::Int

    loss::Float64
    loss_function::Function
    optimiser::Function

    model::Chain
    MLPRegressor(tol, epochs, epochs_nochange, batch_size, batch_shuffle,
                    ninputs, noutputs, loss_function, optimiser, model) =
        new(tol, epochs, epochs_nochange, batch_size, batch_shuffle,
            ninputs, noutputs, 0, loss_function, optimiser, model)
end

function MLPRegressor(layer_sizes=(1, 100, 1); solver, activation=Flux.relu,
                        λ=0.0001, batch_size=-1, shuffle=true, epochs=200,
                        epochs_no_change=10, tol=1e-4)
    # TODO - Check Arguments!

    # Create model chain
    nlayers = length(layer_sizes)
    layers = [Dense(layer_sizes[i], layer_sizes[i+1], activation)
                for i in 1:(nlayers-2)]
    layers = vcat(layers, Dense(layer_sizes[nlayers-1], layer_sizes[nlayers]))
    model = Chain(layers...)

    # Define Loss Function and L2 Regularization
    L2penalty(coeffs) = sum(map(x -> x[:] |> x -> x' * x , coeffs))
    loss_f(x, y) = Flux.mse(model(x), y) + λ * L2penalty(coeffs(model))

    # Define Optimiser
    opt = solver(params(model))
    MLPRegressor(tol, epochs, epochs_no_change, batch_size, shuffle,
                 layer_sizes[1], layer_sizes[end], loss_f, opt, model)
end

# Selectors
batch_size(r::MLPRegressor) = r.batch_size
batch_shuffle(r::MLPRegressor) = r.batch_shuffle

epochs(r::MLPRegressor) = r.epochs

loss(r::MLPRegressor) = r.loss
loss_function(r::MLPRegressor) = r.loss_function
optimiser(r::MLPRegressor) = r.optimiser

# Setters
function set_loss!(r::MLPRegressor, l) =
        r.loss = l
        return nothing
end

# ---------------------------------------------------------------------------
# Training and Prediction Routines
# ---------------------------------------------------------------------------
batch_size(size, nsamples) = size == -1 ? min(200, nsamples) :
size < 1 ? 1 :
size > nsamples ? nsamples :
size

"Groups `n` points in batches of size `bsize`, shuffling them if specified."
function gen_batches(bsize, n, shuffle)
    ixs = shuffle ? Random.shuffle(Vector(1:n)) : 1:n
    [ixs[i:min(i+bsize-1, n)] for i in 1:bsize:n]
end

function mlptrain!(r::MLPRegressor, batches; cb=() -> ())
    batch_loss = 0
    # To register the current loss, save value in batch_loss
    function loss(x, y)
        res = loss_function(r)(x,y)
        batch_loss += Flux.data(res) * size(x, 2)
        res
    end

    Flux.train!(loss, batches, optimiser(r), cb=cb)
    batch_loss
end

function ScikitLearnBase.fit!(lr::MLPRegressor, X, y)
    nsamples = size(X, 2)
    bsize, shuffle = batch_size(batch_size(lr), nsamples), batch_shuffle(lr)

    for epoch in 1:epochs(lr)
        batches = [(X[:,batch_slice], y[:,batch_slice])
                    for batch_slice in gen_batches(bsize, nsamples, shuffle)]
        batches_loss = mlptrain!(lr, batches)  # TODO - use cb to stop iterations
                                               # if max_iter exceed or (tol or n_iter_no_change)
        loss = batches_loss / nsamples

        @info "[$(now())] Epoch: $epoch, loss: $(round(loss, digits=8))"
        set_loss!(lr, loss)
    end
end

ScikitLearnBase.predict(lr::MLPRegressor, X) = Tracker.data(lr.model(X))

#= Tests
X1 = [1 2 3 4 5 6 7 8 9 10;]
y1 = vcat(map(identity,X1), map(x -> -x |> identity, X1))

solver = ADAM()
reg1 = MLPRegressor((1, 100, 100, 2), solver=solver, batch_size=5)
fit!(reg1, X1, y1)
y_pred1 = predict(reg1, X1)

using Plots
scatter(X1', y1', title="MLP Regression", label="Data")
plot!(X1', y_pred1', linewidth=6, label="MLP model (ReLU)")

=#


#= Tests
X2 = [0 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30;
      0 -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 -29 -30;]
y2 = mapslices(x -> x[1] * x[2], X2, dims=1)

solver = ADAM()
reg2 = MLPRegressor((2, 100, 50, 100, 1), solver=solver, batch_size=5, epochs=500, λ=0.1)
fit!(reg2, X2, y2)
y_pred2 = predict(reg2, X2)

using Plots
scatter(y2', X2', title="MLP Regression", label="Data")
plot!(y_pred2', X2', linewidth=6, label="MLP model (ReLU)")
scatter(X2', y2', title="MLP Regression", label="Data")
plot!(X2', y_pred2', linewidth=6, label="MLP model (ReLU)")


L2penalty(coeffs) = sum(map(x -> x[:] |> x -> x' * x , coeffs))
loss_f(x, y) = Flux.mse(reg2.model(x), y2) + λ * L2penalty(coeffs(reg2.model))
loss_f(X2, y2)
=#
