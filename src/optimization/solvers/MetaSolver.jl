# using .Metamodels
using .Pareto: ParetoResult

using DelimitedFiles: readdlm, writedlm
# ------------------------------------------------------------------------
# Solution Converter Routines
# ------------------------------------------------------------------------
convert(::Type{Solution}, x, y, constraints) =
    let variables = convert(typeof_variables(Solution), x)
        objectives = convert(typeof_objectives(Solution), y)

        # Constraints
        if length(constraints) > 0
            constraints = convert(typeof_constraints(Solution), map(c -> evaluate(c, x...), constrs))
            constraint_violation = convert(typeof_constraint_violation(Solution), evaluate_penalty(constrs, x...))

            # Booleans
            feasible = constraint_violation != 0
            evaluated = true

            Solution(variables, objectives, constraints, constraint_violation, feasible, evaluated)
        else
            Solution(variables, objectives)
        end
    end
convert(::Type{Vector{Solution}}, X, y, constraints) =
    map(1:size(X, 2)) do sample
        convert(Solution, X[:, sample], y[:, sample], constraints)
    end

# ------------------------------------------------------------------------
# Creation Routines
# ------------------------------------------------------------------------
"""
There are two main creation approaches: (1) Sampling-based and (2) File-based.
The first one consists on using a sampling method to create multiple
configurations for the parameters and then evaluate each sample using an
`evaluator`. Note that it is also possible to read the samples from a file,
instead of invoking a sampling method.

The second approach consists in loading the whole information from a file, i.e.,
it is not necessary to generate new samples, nor to evaluate them because that
information is already provided out-of-the-box in the specified file.

# Examples
julia>

"""
generate_samples(;  nvars, nsamples, sampling_function, evaluate, unscalers=[],
                    clip::Bool=false, transform=identity, _...) = let
        unscale(V) = for (index, unscaler) in enumerate(unscalers);
                        V[index,:] = unscaler(V[index,:]); nothing
                    end
        clip_it(val, limit) = clip ? min(val, limit) : val

        X = sampling_function(nvars, nsamples); unscale(X);
        X = X[:, 1:clip_it(size(X, 2), nsamples)]
        X = transform(X)
        y = mapslices(evaluate, X, dims=1)
        X, y
    end

store_samples(;filename, header=nothing, dlm=',', gensamples_kwargs...) =
    let (X, y) = generate_samples(;gensamples_kwargs...)
        open(filename, "w") do io
            if header != nothing
                join(io, header, dlm)
                write(io, '\n')
            end
            writedlm(io, vcat(X, y)', dlm)
        end
        X, y
    end
load_samples(;nsamples=Colon, vars_cols, objs_cols, filename, has_header::Bool=true, dlm=',', _...) =
    let data = open(filename, "r") do io
                    has_header ? readline(io) : nothing;
                    readdlm(io, dlm, Float64, '\n')
                end
        X, y = data[nsamples, vars_cols], data[nsamples, objs_cols]
        X', y'
    end

"""
    create_samples(; kwargs...)

Dispatches the sampling routines according to the provided arguments.
There are three main sampling routines:
- [`load_samples`](@ref): given a `filename`, loads the samples from the file.
It is necessary to know which columns refer to the variables and which columns
refer to the objectives, and therefore it requires the `vars_cols` and `objs_cols`
to be specified. If the argument `nsamples` is supplied, then it will return
the first `nsamples` that were loaded from the file `filename`, otherwise it
will return all the samples.

- [`generate_samples`](@ref): given a `sampling_function`, the number of
dimensions `nvars`, and the number of samples `nsamples`, applies the sampling
function to the `nvars` and `nsamples` parameters and obtains a set of samples.
Since not all sampling routines depend on both parameters, if `clipped` is
specified, the number of samples will be clipped, i.e., it will return at most
the specified nsamples. If `clipped` is not specified, then the result of
applying the sampling function will be returned. If `unscalers` are specified
they will unscale the result of the sampling routines. The unscalers should be
an array of unscaling functions receiving a new value, the current minimum and
the current maximum per dimension. It is assumed that the unscaling functions
already possess the knowledge of the variables bounds within the function as
free variables.

- [`store_samples`](@ref): given a `sampling_function` and a `filename` it
first generate samples using the [`generate_samples`](@ref) method and then
stores it in the specified `filename`.

# Arguments
- `nvars::Int`: the number of variables, i.e., dimensions of the samples.
- `nsamples::Int`: the number of samples to be created.
- `sampling::Function`: the sampling function to be used. It must be a function
    receiving two parameters: the number of variables and the number of samples
- `evaluate::Function`: the function that will receive a sample and produce the
    corresponding objective value.
- `unscalers::Vector{Function}=[]`: an array with the unscalers for each variable.
The provided sampling algorithms are unitary, producing samples with ranges in
    the interval [0, 1]. Each unscaler function must receive a value to be unscaled,
    as well as the old minimum and the old maximum. Defaults to empty vector, in
    which case no unscalers will be applied.
- `clip::Bool=false`: variable indicating if we strictly want the specified `nsamples`.
This is necessary as there are many sampling algorithms for which the number of
    samples is exponential in the number of dimensions (`nvars`).
- `transform::Function=identity`: function that allows to apply a transformation to the
    samples that are produced by the sampling algorithm. It receives as argument
    the matrix with the unscaled samples produced by the sampling function.
- `filename::String`: the filename to read samples from or to store samples to.
    Mandatory when reading samples from file.
- `header::Vector{String}=nothing`: the header to insert in the file when storing the
    samples. If not specified, the file will not have header.
- `has_header::Bool=false`: indicator of whether there exists an header in the file
    from which the samples will be read. Defaults to false.
- `dlm::Char=','`: the delimiter of the sample values in the file.
- `vars_cols::Vector{Int}`: the columns corresponding to the variables that will
    be loaded from the file.
- `objs_cols::Vector{Int}`: the columns corresponding to the objectives that
    will be loaded from the file.

"""
create_samples(;kwargs...) =
    if !haskey(kwargs, :sampling_function)
        haskey(kwargs, :filename) ?
            generate_samples(;load_samples...) :
            throw(ArgumentError("invalid sampling methods"))
    else
        λ = kwargs[:sampling_function]
        λ = Sampling.exists(λ) ? Sampling.get_existing(λ; kwargs...) : λ
        haskey(kwargs, :filename) ?
            store_samples(; sampling_function=λ, kwargs...) :
            generate_samples(; sampling_function=λ, kwargs...)
    end

# ---------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------
"""
    Surrogate(type, (objective1, ..., objectiven))
    Surrogate(type, (objective1, ..., objectiven), f, f_params)
    Surrogate(type, (objective1, ..., objectiven), f, f_params, g, g_frequency)
    Surrogate(type, (objective1, ..., objectiven), f, f_params, g, g_frequency, η)

Represents an approximation (or meta) model of another function. Each meta model
possesses the list of objective functions it approximates, as well as the
creation and correction methods they are built and updated with. The correction
frequency is meant to limit the number of corrections the meta model is to be
updated and the exploration_decay_rate (or `η`) is meant to decrease the focus
on global exploration and, consequently, allow the meta model to become more
and more local, i.e., to exploit locality and promising regions.

# Examples
julia> sampling_creation_params = Dict{Symbol, Any} {
    :sampling_function => stratifiedMC,
    :bins => [3, 4],
    :nsamples => 30,
    :filename => "sMC-sample.csv",
    :header => ["Var1", "Var2", "Obj1", "Obj2"]
    :dlm => ',',
};

julia> file_creation_params = Dict{Symbol, Any} {
    :filename => "sMC-sample.csv",
    :dlm => ',',
    :has_header => true,
    :vars => [1, 2, 3],
    :objs => [1, 2, 3]
};

"""
struct Surrogate
    meta_model::Any
    objectives::Tuple{Vararg{AbstractObjective}}

    objectives_indices::Union{Colon, Vector{Int}}
    variables_indices::Union{Colon, Vector{Int}}

    # Surrogates can be loaded from files or by sampling
    creation::Function
    creation_params::Dict{Symbol, Any}

    # Surrogates can be updated from t to t
    correction::Function
    correction_frequency::Real

    # Surrogates evaluate samples
    evaluation::Function

    # Surrogates should increase exploitation with the increase of evaluations
    # TODO - Adaptively change the surrogate to give more weight to newly
    # obtained data, then to older one.
    exploration_decay_rate::Real

    Surrogate(meta_model; objectives::Tuple{Vararg{AbstractObjective}},
              objectives_indices=(:), variables_indices=(:),
              creation_f::Function=Metamodels.fit!, creation_params::Dict{Symbol, Any}=Dict{Symbol, Any}(),
              correction_f::Function=Metamodels.fit!, correction_frequency::Int=1,
              evaluation_f::Function=Metamodels.predict, decay_rate::Real=0.1) =
        begin
            if isempty(objectives)
                throw(DomainError("invalid argument `objectives` cannot be empty."))
            elseif correction_frequency < 0
                throw(DomainError("invalid argument `correction_frequency` must be a non-negative integer."))
            elseif decay_rate < 0
                throw(DomainError("invalid argument `decay_rate` must be a non-negative real."))
            end
            new(meta_model, objectives, objectives_indices, variables_indices, creation_f, creation_params, correction_f, correction_frequency, evaluation_f, decay_rate)
        end
end

# Selectors
@inline sampling_transform(s::Surrogate) =
(V) -> let  surrogate_vars = variable_indices(s)
            V_temp = zeros(maximum(surrogate_vars), size(V, 2))
            V_temp[surrogate_vars] = V
            V_temp
end

@inline surrogate(s::Surrogate) = s.meta_model
@inline objectives(s::Surrogate) = s.objectives

@inline objectives(s::Surrogate, y) = y[s.objectives_indices,:]
@inline variables(s::Surrogate, X) = X[s.variables_indices,:]

@inline variables_indices(s::Surrogate) = s.variables_indices
@inline objectives_indices(s::Surrogate) = s.objectives_indices
@inline nobjectives(s::Surrogate) = sum(map(nobjectives, objectives(s)))

@inline correction_function(s::Surrogate) = s.correction
@inline correction_function(s::Surrogate, X, y) =
    s.correction(surrogate(s), X, y)

@inline creation_function(s::Surrogate) = s.creation
@inline creation_function(s::Surrogate, X, y) =
    s.creation(surrogate(s), X, y)

@inline evaluation_function(s::Surrogate) = s.evaluation
@inline evaluation_function(s::Surrogate, X) = s.evaluation(s.meta_model, variables(s, X))

@inline creation_params(s::Surrogate) = s.creation_params
@inline creation_param(s::Surrogate, param::Symbol, default) =
    get(creation_params(s), param, default)

# Predicate
@inline is_multi_target(s::Surrogate) = nobjectives(s) > 1

# Modifiers
train!(surrogates::Vector{Surrogate}, X, y) =
    foreach(surrogate ->
                creation_function(  surrogate,
                                    objectives(surrogate, X),
                                    variables(surrogate, y)),
            surrogates)

train!(surrogate::Surrogate, model=nothing) =
    let evaluate(x...) = map(o -> apply(o, x...), objectives(surrogate)) |> flatten
        params = creation_params(surrogate)
        surrogate_vars = variables_indices(surrogate)

        # Retrieve unscalers from model if user does not specify unscalers
        if model != nothing && !haskey(params, :unscalers)
            params[:unscalers] = unscalers(model)[surrogate_vars]
        end

        # If special case of surrogate, specify extra configurations (override old ones if necessary)
        if surrogate_vars != Colon
            params[:nvars] = length(surrogate_vars)
            params[:transform] = sampling_transform(s)
        end
        X, y = create_samples(;params...)
        creation_function(surrogate, objectives(surrogate, X), variables(surrogate, y))
        X, y
    end
correct!(surrogate::Surrogate, data::Vector{Solution}) =
    let nsols = length(data)
        X = hcat(map(variables, data)...)[variables_indices(surrogate),:]
        y = hcat(map(objectives, data)...)[objectives_indices(surrogate),:]

        # Fit surrogate
        correction_function(surrogate, X, y)
        # TODO - Get error measurement
        0
    end

export Surrogate
# ---------------------------------------------------------------------
# MetaModel (MetaProblem)
# TODO - This is the same as the Model - define macro?
# ---------------------------------------------------------------------
"""
    MetaModel(variables, objectives, constraints)

Represents a problem where the objectives are approximated by other, usually
cheaper, model(s) called surrogates. From a MetaModel we are able to derive
two different models:
    - Cheaper model, where the objectives of the [`Model`](@ref) are the
    surrogate functions.
    - Expensive model, where the objectives of the [`Model`](@ref) are the
    real/true objective functions (e.g., simulations).
"""
struct MetaModel <: AbstractModel
    variables::Vector{AbstractVariable}
    objectives::Vector{Surrogate}
    constraints::Vector{Constraint}

    MetaModel(vars::Vector{T}, objs::Vector{Surrogate}, constrs::Vector{Constraint}=Vector{Constraint}()) where{T<:AbstractVariable} =
        new(vars, objs, constrs)
end

# Selectors
surrogates(m::MetaModel) = objectives(m)
unsafe_surrogates(m::MetaModel) = unsafe_objectives(m)
original_objectives(m::MetaModel) = flatten(map(objectives, surrogates(m)))

nobjectives(m::MetaModel) = sum(map(nobjectives, m.objectives))

# Create different Models from MetaModel
cheap_model(m::MetaModel; dynamic::Bool=false) =
    let surrogates = dynamic ? unsafe_surrogates : surrogates
        vars = variables(m)
        constrs = constraints(m)
        objs = map(surrogates(m)) do surrogate
            λ = (x...) -> evaluation_function(surrogate, reshape(x..., (length(vars), 1)))

            coeffs = foldl(vcat, map(coefficient, objectives(surrogate)), init=Real[]) # TODO - Broke abstraction barrier! FIX it later
            snses = foldl(vcat, map(sense, objectives(surrogate)), init=Symbol[])

            # Create Objective
            is_multi_target(surrogate) ? SharedObjective(λ, coeffs, snses) :
                                 Objective(λ, coeffs..., snses...)
        end
        Model(vars, objs, constrs)
    end
expensive_model(m::MetaModel) =
    Model(variables(m), original_objectives(m), constraints(m))

export MetaModel
# ---------------------------------------------------------------------
# MetaSolver
# ---------------------------------------------------------------------
"""
    MetaSolver(solver, nvariables, nobjectives, max_eval)

Concretization of [`AbstractSolver`](@ref), which maintains a solver, the
maximum number of expensive evaluations to be reached in the optimization
process and the results.

The solver will be responsible for exploiting the surrogates in the search for
candidate optimal solutions, which are then subject of expensive evaluations to
assert their real value. This information is then stored in the results and
used to update the surrogates. The MetaSolver will stop when a termination
condition is met.
"""
mutable struct MetaSolver <: AbstractSolver
    solver::AbstractSolver

    max_evaluations::Int
    sampling_params::Dict{Symbol, Any}

    pareto_result::ParetoResult
    MetaSolver(solver; nvars, nobjs, sampling_params, max_eval=100) = begin
            if max_eval < 0
                throw(DomainError("invalid argument `max_eval` must be a positive integer"))
            elseif isempty(sampling_params)
                throw(DomainError("invalid argument `sampling_params` must provide
                enough parameters to run the initialization routine for the surrogates"))
            # Ready-to-use data is specified (no sampling necessary)
            elseif haskey(sampling_params, :X) && !haskey(sampling_params, :y) ||
                   !haskey(sampling_params, :X) && haskey(sampling_params, :y)
                throw(DomainError("invalid argument `sampling_params` must provide
                both `:X` and `:y` in order to run the initialization routine for the surrogates"))
            end
            new(solver, max_eval, sampling_params, ParetoResult(nvars, nobjs))
        end
end

# Selectors
"Returns the number of maximum expensive evaluations to run the optimisation"
max_evaluations(s::MetaSolver) = s.max_evaluations
"Returns the solver responsible for exploring the cheaper models"
optimiser(s::MetaSolver) = s.solver
"Returns the Pareto Results"
results(s::MetaSolver) = s.pareto_result
"Returns the Meta Solver initial sampling params"
sampling_params(s::MetaSolver) = s.sampling_params
sampling_data(s::MetaSolver) = !is_sampling_required(s) ?
        (s.sampling_params[:X], s.sampling_params[:y]) :
        throw(DomainError("invalid operation. The specified MetaSolver does not
        have ready-to-use data `X` and `y`"))

# Predicates
is_sampling_required(s::MetaSolver) =
    !haskey(s.sampling_params, :X) || !haskey(s.sampling_params, :y)

"Returns the Pareto Front solutions obtained with the MetaSolver"
ParetoFront(s::MetaSolver) = Pareto.ParetoFront(results(s))

# Modifiers
"Stores the variables and objectives in the corresponding meta solver"
Base.push!(solver::MetaSolver, variables, objectives) =
    let nvars = size(variables, 2)
        nobjs = size(objectives, 2)
        rsults = results(solver)
        if nvars == nobjs
            foreach(n -> push!(rsults, variables[:, n], objectives[:, n]), 1:nobjs)
        else
            throw(DimensionMismatch("the number of variables and objectives does not match"))
        end
    end
Base.push!(solver::MetaSolver, solutions::Vector{Solution}) =
    let vars = hcat(map(variables, solutions)...)
        objs = hcat(map(objectives, solutions)...)
        push!(solver, vars, objs)
    end

# Solve -------------------------------------------------------------------
"Clips the number of elements in `elements` to be at most `nmax` or if there
fewer elements than `nmax` then it returns all elements and `nmax` - `#elements`"
clip(elements, nmax) = let
    elements = length(elements) > nmax ? elements[1:nmax] : elements
    elements, nmax - length(elements)
    end

solve(meta_solver::MetaSolver, meta_model::MetaModel) =
    let solver = optimiser(meta_solver)
        evals_left = max_evaluations(meta_solver)
        correct(solutions) = surrogate ->
                                let err = correct!(surrogate, solutions)
                                    @info "[$(now())] Retrained surrogate exhibits $(err)% error."
                                    err
                                end

        # Models
        cheaper_model = cheap_model(meta_model, dynamic=true)
        expensiv_model = expensive_model(meta_model)

        # Step 1. Create each Surrogate
        X, y = [], []
        if is_sampling_required(meta_solver)
            @info "[$(now())] Creating samples..."
            X, y = create_samples(; nvars=nvariables(cheaper_model),
                                    evaluate=(x) -> evaluate(expensiv_model, x) |> objectives,
                                    unscalers=unscalers(expensiv_model),
                                    sampling_params(meta_solver)...)
        else
            @info "[$(now())] Loading provided samples..."
            X, y = sampling_data(solver)
        end

        @info "[$(now())] Initializing surrogates with samples..."
        train!(unsafe_surrogates(meta_model), X, y)
        @info "[$(now())] Pushing sampling data to Pareto Front..."
        push!(meta_solver, X, y)

        # Repeat until termination condition is met.
        while evals_left > 0
            # Step 2. Apply optimizer to cheap model
            candidate_solutions = solve(solver, cheaper_model)

            # Step 3. Evaluate solutions from 2, using the expensive model
            # Guarantee that the number of Max Evals is satisfied
            candidate_solutions, evals_left = clip(candidate_solutions, evals_left)
            @info "[$(now())] Found $(length(candidate_solutions)) candidate solutions... \n\tExpensive evaluations left: $evals_left"
            solutions = evaluate(expensiv_model, candidate_solutions)

            # Step 4. Add results from Step 3. to ParetoResult
            push!(meta_solver, solutions)
            # Step 5. Update the surrogates
            foreach(correct(solutions), unsafe_surrogates(meta_model))
        end
        # Step 7. Return Pareto Result non-dominated
        convert(Vector{Solution}, ParetoFront(meta_solver)..., constraints(expensiv_model))
    end

export MetaSolver, solve
