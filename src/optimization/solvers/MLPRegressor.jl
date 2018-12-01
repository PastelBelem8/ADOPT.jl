using Flux
using Dates
using Random

# ---------------------------------------------------------------------------
# Training Utils: train_test_split, batches_split
# ---------------------------------------------------------------------------
train_test_split(X, y, test_fraction=0.25, shuffle=false) =
    let (ndims, nsamples) = size(X); test_size = trunc(Int, nsamples * test_fraction)
        ixs = shuffle ? Random.shuffle(Vector(1:nsamples)) : Vector(1:nsamples)

        # X_Train, X_Test, y_Train, y_Test
        X[:, ixs[test_size+1:end]], test_size == 0 ? zeros(ndims, 0) : X[:, ixs[1:test_size]],
        y[:, ixs[test_size+1:end]], test_size == 0 ? zeros(ndims, 0) : y[:, ixs[1:test_size]]
    end

"Groups `n` points in batches of size `bsize`, shuffling them if specified."
gen_batches(bsize, n, shuffle=false) =
    let ixs = shuffle ? Random.shuffle(Vector(1:n)) : Vector(1:n)
        [ixs[i:min(i+bsize-1, n)] for i in 1:bsize:n]
    end

"Computes the l2 penalty for matrix `X`"
L2penalty(X::T) where{T<:AbstractArray} = let x = X[:]; sum(x' * x) end
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

    EarlyStopping(tol=1e-4, epochs_nochange=10, validation_fraction=0.1) =
        begin
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

update!(es::EarlyStopping, X_val, y_val, λloss, params) =
    let loss = λloss(X_val, y_val)
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
# call(f, xs...) = f(xs...)

# Getters  -----------------------------------------------------------------
get_params(m::Chain; weights::Bool) =
    let i = weights ? 1 : 2;
        map(Tracker.data, Flux.params(m))[i:2:end]
    end
weights(m::Chain) = get_params(m, weights=true)
bias(m::Chain) = get_params(m, weights=false)

"Stacks multiple layers with sizes `layer_sizes` and except for the last
layer, assigns them the specified `activation` function."
Chain_by_sizes(activation, layer_sizes...) =
    let create_layer(in, out) = Dense(layer_sizes[in], layer_sizes[out], activation)
        layers = vcat(map(i -> create_layer(i, i+1), 1:(length(layer_sizes)-2)),
                      Dense(layer_sizes[end-1], layer_sizes[end]))

        Flux.Chain(layers...)
    end

# ---------------------------------------------------------------------------
# MLPRegressor
# ---------------------------------------------------------------------------
FluxOptimisers = Union{Flux.ADAM, Flux.ADADelta, Flux.ADAGrad, Flux.AMSGrad, Flux.NADAM, Flux.Nesterov, Flux.Optimiser, Flux.RMSProp}

mutable struct MLPRegressor
    epochs::Int
    batch_size::Int
    shuffle::Bool

    loss::Function
    optimiser::FluxOptimisers

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

MLPRegressor(layer_sizes=(1, 100, 1); solver, activation=Flux.relu, λ=0.0001,
             batch_size=-1, shuffle=true, epochs=200, early_stop=true,
             epochs_nochange=10, tol=1e-4, val_fraction=0.10) = let
        # Create Early Stopping
        earlystop = EarlyStopping(tol, epochs_nochange, early_stop ? val_fraction : 0)

        # Create Network
        model = Chain_by_sizes(activation, layer_sizes...)

        # Create loss_function w/ L2 regularization
        loss(x, y) = Flux.mse(model(x), y) + λ * L2penalty(weights(model))
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
Random.shuffle(r::MLPRegressor) = r.shuffle

early_stopping(r::MLPRegressor) = r.early_stopping
epochs(r::MLPRegressor) = r.epochs

loss_curve(r::MLPRegressor) = r.losses
loss_function(r::MLPRegressor) = r.loss

params(r::MLPRegressor; flux::Bool=false) =
    flux ? Flux.params(r.model) : map(Tracker.data, Flux.params(r.model))
params(r::MLPRegressor) = map(Tracker.data, Flux.params(r.model))
optimiser(r::MLPRegressor) = r.optimiser

# Modifiers ------------------------------------------------------------------
update_losses!(r::MLPRegressor, loss) = push!(r.losses, loss);
update_params!(r::MLPRegressor, params) =
    begin
        @warn "[$(now())] Updating MLP Regressor Parameters"
        @info "[Before] MLP Regressor Parameters: $((map(Tracker.data, params(r.model)))[:])"
        new_layers = map(layer -> Dense(layer.W, layer.b, layer.σ), r.model.layers)
        r.model = Chain(new_layers...)
        @info "[After] MLP Regressor Parameters: $((map(Tracker.data, params(r.model)))[:])"
    end

# Training and Prediction Routines -----------------------------------------
"Creates the loss function to be used for batch training."
batches_loss(r::MLPRegressor; cb=() -> ()) =
    let accumulated_loss = 0
        function loss_per_batch(x, y)
            res = loss_function(r)(x,y)
            accumulated_loss += Flux.data(res) * size(x, 2)
            res
        end

        (batches) -> begin
            accumulated_loss = 0
            Flux.train!(loss_per_batch, params(r, flux=true), batches, optimiser(r), cb=cb);
            # foreach(call, cb)
            accumulated_loss
        end
    end

mlptrain!(r::MLPRegressor, X, y; cb=() -> ()) =
    let batch_train = batches_loss(r,  cb=cb);
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

ScikitLearnBase.fit!(r::MLPRegressor, X, y) = begin mlptrain!(r, X, y); return r end
ScikitLearnBase.predict(r::MLPRegressor, X) = mlppredict(r, X)
