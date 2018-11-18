using Flux
using Dates
using Random

using ScikitLearnBase
import ScikitLearnBase: fit!, predict

# ---------------------------------------------------------------------------
# Training Utils: train_test_split, batches_split
# ---------------------------------------------------------------------------
function train_test_split(X, y, test_fraction=0.25, shuffle=false)
    ndims, nsamples = size(X); test_size = trunc(Int, nsamples * test_fraction)
    ixs = shuffle ? Random.shuffle(Vector(1:nsamples)) : Vector(1:nsamples)

    # X_Train, X_Test, y_Train, y_Test
    X[:, ixs[test_size+1:end]], test_size == 0 ? zeros(ndims, 0) : X[:, ixs[1:test_size]],
    y[:, ixs[test_size+1:end]], test_size == 0 ? zeros(ndims, 0) : y[:, ixs[1:test_size]]
end

"Groups `n` points in batches of size `bsize`, shuffling them if specified."
function gen_batches(bsize, n, shuffle=false)
    ixs = shuffle ? Random.shuffle(Vector(1:n)) : Vector(1:n)
    [ixs[i:min(i+bsize-1, n)] for i in 1:bsize:n]
end

"Computes the l2 penalty for matrix `X`"
L2penalty(X::T)         where{T<:AbstractArray} =
    let x = X[:]; sum(x' * x) end
L2penalty(X::Vector{T}) where{T<:AbstractArray} = sum(map(L2penalty, X))

# ---------------------------------------------------------------------------
# Validation-Based Early Stopping
# ---------------------------------------------------------------------------
mutable struct EarlyStopping
    tolerance::Float64
    epochs_nochange::Int
    epochs_nochange_count::Int

    validation_fraction::Float64
    validation_losses::Vector{Real}

    best_loss::Float64
    best_params::Vector

    function EarlyStopping(tol=1e-4, epochs_nochange=10, validation_fraction=0.1)
        @assert tol > 0 "Tolerance cannot be a negative value"
        @assert epochs_nochange > 0 "Number of epochs cannot be a negative value"
        @assert 0 ≤ validation_fraction ≤ 1 "The validation set's fraction must be within 0 and 1."

        new(tol, epochs_nochange, 0, validation_fraction, Vector{Real}(), Inf, Vector())
    end
end

# Selectors ----------------------------------------------------------------
best_loss(es::EarlyStopping) = es.best_loss
best_params(es::EarlyStopping) = es.best_params

nochange(es::EarlyStopping) = es.epochs_nochange
nochange_count(es::EarlyStopping) = es.epochs_nochange_count

tolerance(es::EarlyStopping) = es.tolerance
validation_fraction(es::EarlyStopping) = es.validation_fraction


# Predicates ----------------------------------------------------------------
is_improvement(es::EarlyStopping, loss::Real) =
    loss < best_loss(es) + tolerance(es)
is_toStop(es::EarlyStopping) =
    nochange(es) == nochange_count(es)
is_validation_based(es::EarlyStopping) = validation_fraction(es) > 0

# Modifiers -----------------------------------------------------------------
update_loss!(es::EarlyStopping, loss::Real) =
    push!(es.validation_losses, loss)
update_epochs_nochange!(es::EarlyStopping, loss::Real) =
    es.epochs_nochange_count = is_improvement(es, loss) ?  0 : nochange_count(es) + 1
update_best!(es::EarlyStopping, loss, params) =
    if loss < best_loss(es)
        es.best_loss = loss
        es.best_params = deepcopy(params)
    end

function update!(es::EarlyStopping, X_val, y_val, λloss, params)
    loss = λloss(X_val, y_val)
    @info "[$(now())] Early Stopping loss: $(round(loss, digits=8))"
    update_loss!(es, loss)
    update_epochs_nochange!(es, loss)
    update_best!(es, loss, params)
end
# Others
status(es::EarlyStopping) =
    "$(is_validation_based(es) ? "Validation" : "Training") loss did not improve more than $(tolerance(es)) for $(nochange(es)) consecutive epochs."

train_test_split(es::EarlyStopping, X, y, shuffle) =
    train_test_split(X, y, validation_fraction(es), shuffle)

# ---------------------------------------------------------------------------
# Flux Utils
# ---------------------------------------------------------------------------
call(f, xs...) = f(xs...)

# Flux Optimisers -----------------------------------------------------------
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

# Getters  -----------------------------------------------------------------
get_params(m::Chain; weights::Bool) = let
    i = weights ? 1 : 2;
    map(Tracker.data, Flux.params(m)[i:2:end])
end
weights(m::Chain) = get_params(m, weights=true)
bias(m::Chain) = get_params(m, weights=false)


"Stacks multiple layers with sizes `layer_sizes` and except for the last
layer, assigns them the specified `activation` function."
function Chain_by_sizes(activation, layer_sizes...)
    create_layer(in, out) = Dense(layer_sizes[in], layer_sizes[out], activation)

    layers = vcat(map(i -> create_layer(i, i+1), 1:(length(layer_sizes)-2)),
                  Dense(layer_sizes[end-1], layer_sizes[end]))

    Flux.Chain(layers...)
end

# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------
mutable struct MLPRegressor
    epochs::Int
    batch_size::Int
    shuffle::Bool

    loss::Function
    optimiser::Function

    model::Chain
    losses::Vector{Real}
    early_stopping::EarlyStopping

    # Constructor
    MLPRegressor(epochs, batch_size, shuffle, loss_f, opt, model, early_stop::EarlyStopping) =
        new(epochs, batch_size, shuffle, loss_f, opt, model, Vector{Real}(),
            early_stop)
    MLPRegressor(epochs, batch_size, shuffle, loss_f, opt, model, tol=1e-4,
                    epochs_nochange=10, val_fraction=0.1) =
        MLPRegressor(epochs, batch_size, shuffle, loss_f, opt, model,
            EarlyStopping(tol, epochs_nochange, val_fraction))
end

function MLPRegressor(layer_sizes=(1, 100, 1); solver, activation=Flux.relu,
                        λ=0.0001, batch_size=-1, shuffle=true, epochs=200,
                        early_stop=true, epochs_nochange=10, tol=1e-4, val_fraction=0.10)
    # Create Early Stopping
    earlystop = EarlyStopping(tol, epochs_nochange, early_stop ? val_fraction : 0)

    # Create Network
    model = Chain_by_sizes(activation, layer_sizes...)

    # Create loss_function w/ L2 regularization
    loss(x, y) = Flux.mse(model(x), y) + λ * L2penalty(weights(model))

    solver = solver(Flux.params(model))
    MLPRegressor(epochs, batch_size, shuffle, loss, solver, model, earlystop)
end

"Calculates `batch_size`, given the `max size` and the proposed `bsize`.
Clipping the value of `bsize` if larger than `maxsize` or smaller than 1. If
no `bsize` is specified, the batch size will be 200 or `maxsize` if smaller
than 200."
batch_size(maxsize, bsize=-1) = bsize == -1 ?
min(200, maxsize) :
bsize < 1 ? 1 :
bsize > maxsize ? maxsize :
bsize

# Selectors ----------------------------------------------------------------
batch_size(r::MLPRegressor) = r.batch_size
shuffle(r::MLPRegressor) = r.shuffle

early_stopping(r::MLPRegressor) = r.early_stopping
epochs(r::MLPRegressor) = r.epochs

loss_curve(r::MLPRegressor) = r.losses
loss_function(r::MLPRegressor) = r.loss

params(r::MLPRegressor) = map(Tracker.data, Flux.params(r.model))
optimiser(r::MLPRegressor) = r.optimiser

# Modifiers ------------------------------------------------------------------
update_losses!(r::MLPRegressor, loss) = push!(r.losses, loss);
function update_params!(r::MLPRegressor, params)
    @warn "[$(now())] Updating MLP Regressor Parameters"
    @debug "[Before] MLP Regressor Parameters: $(Tracker.data(params(r.model)))"
    new_layers = map(layer -> Dense(layer.W, layer.b, layer.σ), r.model.layers)
    r.model = Chain(new_layers...)
    @debug "[After] MLP Regressor Parameters: $(Tracker.data(params(r.model)))"
end

# Training and Prediction Routines -----------------------------------------
"Creates the loss function to be used for batch training."
function batches_loss(r::MLPRegressor; cb=() -> ())
    accumulated_loss = 0
    function loss_per_batch(x, y)
        res = loss_function(r)(x,y)
        accumulated_loss += Flux.data(res) * size(x, 2)
        res
    end

    (batches) -> begin
        accumulated_loss = 0
        Flux.train!(loss_per_batch, batches, optimiser(r), cb=cb);
        # foreach(call, cb)
        accumulated_loss
    end
end

function mlptrain!(r::MLPRegressor, X, y; cb=() -> ())
    batch_train = batches_loss(r,  cb=cb);
    loss_f = (x, y) -> loss_function(r)(x, y) |> Tracker.data

    shffle = shuffle(r)
    early_stop = early_stopping(r)
    X, X_val, y, y_val = train_test_split(early_stop, X, y, shffle)

    if !is_validation_based(early_stop)
        X_val, y_val = X, y
    end

    nsamples = size(X, 2)
    bsize = batch_size(nsamples, batch_size(r))

    for epoch in 1:epochs(r)
        batches = map(gen_batches(bsize, nsamples)) do batch
                        (X[:,batch], y[:,batch]) end
        loss = batch_train(batches) / nsamples
        @info "[$(now())] Epoch: $epoch, loss: $(round(loss, digits=8))"

        update_losses!(r, loss)
        update!(early_stop, X_val, y_val, loss_f, params(r))

        if is_toStop(early_stop)
            @info status(early_stop)
            update_params!(r, best_params(early_stop))
            return r
        end
    end

    # Optimization has not yet converged
    @warn "Optimizer: Maximum epochs $(epochs(r)) reached and the optimization hasn't converged yet."

end
mlppredict(r::MLPRegressor, X) = Tracker.data(r.model(X))

# Scikit Learn compatibility
ScikitLearnBase.fit!(r::MLPRegressor, X, y) = begin mlptrain!(r, X, y); return r end
ScikitLearnBase.predict(r::MLPRegressor, X) = mlppredict(r, X)


#= Tests
X1 = reshape(Vector(1:1000), (1, 1000))
y1 = vcat(map(identity,X1), map(x -> -x |> identity, X1))

λ=0.01
solver = ADAM()
reg1 = MLPRegressor((1, 100, 100, 2), solver=solver, batch_size=75, epochs=30, λ=λ)

fit!(reg1, X1, y1)

using Plots
plot(Vector(1:length(reg1.early_stopping.validation_losses)), reg1.early_stopping.validation_losses, yscale=:log10, label="Validation Losses")
plot!(Vector(1:length(reg1.losses)), reg1.losses, yscale=:log10, label="Training Losses")

y_pred1 = predict(reg1, X1)

println("Minimum training losses: $(minimum(reg1.losses))")
println("Minimum validation losses: $(minimum(reg1.early_stopping.validation_losses))")

using Plots
scatter(X1', y1', title="MLP Regression", label="Data")
plot!(X1', y_pred1', linewidth=6, label="MLP model (ReLU)")
=#


#= Tests
X2 = [0 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30;
      0 -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 -29 -30;]
y2 = mapslices(x -> x[1] * x[2], X2, dims=1)

λ = 0.03
solver = ADAM(0.3, decay=0.25)
reg2 = MLPRegressor((2, 100, 100, 1), solver=solver, epochs=1200, batch_size=5, λ=λ)
fit!(reg2, X2, y2)
y_pred2 = predict(reg2, X2)

using Plots
scatter(y2', X2', title="MLP Regression", label="Data")
plot!(y_pred2', X2', linewidth=6, label="MLP model (ReLU)")
scatter(X2', y2', title="MLP Regression", label="Data")
plot!(X2', y_pred2', linewidth=6, label="MLP model (ReLU)")

minimum(reg2.losses)

plot(Vector(1:length(reg1.losses)), reg1.losses)
plot!(Vector(1:length(reg1.early_stopping.validation_losses)), reg1.early_stopping.validation_losses)
plot!(Vector(1:length(reg2.losses)), reg2.losses)
plot!(Vector(1:length(reg2.early_stopping.validation_losses)), reg2.early_stopping.validation_losses)
=#
