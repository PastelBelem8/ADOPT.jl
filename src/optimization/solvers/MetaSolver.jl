using .Metamodels
using .Pareto: ParetoResult

# ------------------------------------------------------------------------
# Solution Converter Routines
# ------------------------------------------------------------------------
convert(::Type{Solution}, x, y, constrs) =
    let variables = convert(typeof_variables(Solution), x)
        objectives = convert(typeof_objectives(Solution), y)

        # Constraints
        constraints = convert(typeof_constraints(Solution), map(c -> evaluate(c, x...), constrs))
        constraint_violation = convert(typeof_constraint_violation(Solution), evaluate_penalty(constrs, x...))

        # Booleans
        feasible = (constraint_violation == 0)
        evaluated = true

        Solution(variables, objectives, constraints, constraint_violation, feasible, evaluated)
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
sample(;nsamples, sampling, unscalers, evaluate) =
    let nvars = length()
        X = sampling(nvars, nsamples)
        foreach((unscale, var) -> X[var,:] = unscale(X[var,:], 0, 1), unscalers, 1:nvars)
        X, evaluate(X)
    end

sample_to_file(;nsamples, sampling, unscalers, evaluate, filename, header=nothing, dlm=',') =
    let X, y = sample(nsamples=nsamples, sampling=sampling, unscalers=unscalers, evaluate=evaluate)
        open(filename, "w") do io
            if header != nothing
                write(io, header)
            end
            writeldm(io, [X y], ',')
        end
        X, y
    end

from_file(;nvars::Int=0, vars, objs, filename, has_header::Bool=true, dlm=',', _...) =
    let read(io) = readdlm(io, dlm) ∘ (has_header ? readline : identity)
        data = read(filename)
        X, y = data[:, vars], data[:, (nvars .+ objs)]
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

# Arguments


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
    meta_model::Type
    objectives::Tuple{AbstractObjective}

    # Surrogates can be loaded from files or by sampling
    creation_function::Function
    creation_params::Dict{Symbol, Any}

    # Surrogates can be updated from t to t
    correction_function::Function
    correction_frequency::Real

    # Surrogates should increase exploitation with the increase of evaluations
    # TODO - Adaptively change the surrogate to give more weight to newly
    # obtained data, then to older one.
    exploration_decay_rate::Real

    Surrogate(meta_model; objectives::Tuple{Objectives}, creation_f::Function=fit!,
              creation_params::Dict{Symbol, Any}, correction_f::Function=predict,
              correction_frequency::Int=1, decay_rate::Real=0.1) = begin
        if isempty(objectives)
            throw(DomainError("invalid argument `objectives` cannot be empty."))
        elseif correction_frequency < 0
            throw(DomainError("invalid argument `correction_frequency` must be a non-negative integer."))
        elseif decay_rate < 0
            throw(DomainError("invalid argument `decay_rate` must be a non-negative real."))
        end
        new(meta_model, objectives, creation_f, creation_params, correction_f,
            correction_frequency, decay_rate)
    end
end

# Selectors
surrogate(s::Surrogate) = s.meta_model
objectives(s::Surrogate) = s.objectives

nobjectives(s::Surrogate) = sum(map(nobjectives, objectives(s)))

correction_function(s::Surrogate) = s.correction_function
creation_function(s::Surrogate) = s.creation_function
creation_params(s::Surrogate) = s.creation_params
creation_param(s::Surrogate, param::Symbol, default=nothing) = get(creation_params(s), param, default)

creation_approach(s::Surrogate) = try
    λ = creation_param(s, "sampling_function", throw(DomainError()))
    λparams = creation_params(s)
    sampling = Sampling.exists(λ) ? Sampling.get_existing(λ; λparams...) : λ

    evaluate(x...) = map(o -> apply(o, x...), objectives(s)) |> flatten
    filename = creation_param(s, :filename, "sample-$(string(λ))-$(now()).csv")

    (unscalers) -> sample_to_file(; λparams...,
                                    sampling=λ,
                                    evaluate=evaluate,
                                    unscalers=unscalers,
                                    filename=filename)
    catch y
        isa(y, DomainError) ? (_...) -> from_file(;creation_params(s)...) : throw(y)
    end

# Predicate
is_multi_target(s::Surrogate) = nobjectives(s) > 1

# Modifiers
create!(surrogate::Surrogate, unscalers) =
    let approach = creation_approach(surrogate);
        @info "[$(now())] Creating surrogate using the $(string(approach)) approach";
        X, y = approach(unscalers);

        @info "[$(now())] Training surrogate model for $(size(X, 2)) samples";
        creation_function(surrogate, X, y);
    end

correct!(surrogate::Surrogate, data) = let
    X, y = data
    # Fit surrogate
    correction_function(surrogate, X, y)

    # TODO - Get error measurement
    surrogate, 0
    end

# ---------------------------------------------------------------------
# MetaModel (MetaProblem)
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
# TODO - This is the same as the Model - define macro?
    variables::Vector{AbstractVariable}
    objectives::Vector{Surrogate}
    constraints::Vector{Constraint}
end

# Selectors
surrogates(m::MetaModel) = map(surrogate, objectives(m))
original_objectives(m::MetaModel) = flatten(map(objectives, objectives(m))

# Create different Models from MetaModel
cheap_model(m::MetaModel) =
    let vars = variables(m)
        constrs = constraints(m)
        objs = map(objectives(m)) do surrogate
            λ = X -> predict(surrogate, X)
            coeffs = flatten(map(coefficient, objectives(surrogate)))
            senses = flatten(map(senses, objectives(surrogate)))

            # Create Objective
            obj = is_multi_target(surrogate) ? SharedObjective : Objective
            obj(λ, coeffs, senses)
        end
        Model(vars, objs, constrs)
    end
expensive_model(m::MetaModel) =
    Model(variables(m), original_objectives(m), constraints(m))

# ---------------------------------------------------------------------
# MetaSolver
# ---------------------------------------------------------------------
"""
    MetaSolver(solver, max_eval, nvariables, nobjectives)

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
    pareto_result::ParetoResult
    end

    MetaSolver(solver, nvars, nobjs, max_eval=100) =
        begin
            if max_eval < 0
                throw(DomainError("invalid argument `max_eval` must be a positive integer"))
            end
            new(solver, max_eval, ParetoResult(nvars, nobjs))
        end
end

# Selectors
"Returns the number of maximum expensive evaluations to run the optimisation"
max_evaluations(s::MetaSolver) = s.max_evaluations
"Returns the solver responsible for exploring the cheaper models"
optimiser(s::MetaSolver) = s.solver
"Returns the Pareto Results"
results(s::MetaSolver) = s.pareto_result
"Returns the data stored in the meta solver in the format `(X, y)`"
data(s::MetaSolver) = let
    dt = results(s)
    variables(dt), objectives(dt) # X, y
    end
"Returns the Pareto Front solutions obtained with the MetaSolver"
ParetoFront(s::MetaSolver) = ParetoFront(results(s))

# Modifiers
"Stores the variables and objectives in the corresponding meta solver"
push!(solver::MetaSolver, solutions::Vector{Solution}) =
    foreach(solution -> push!(results(solver), variables(solution), objectives(solution)),
        solutions)

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
        correct(s) -> let err = correct!(s, data(meta_solver))
                          @info "[$(now())] Retrained surrogate exhibits $(err)% error."
                      end

        # Models
        cheaper_model = cheap_model(meta_model)
        expensiv_model = expensive_model(meta_model)

        # Step 1. Create each Surrogate
        foreach(create!, objectives(meta_model))

        # Repeat until termination condition is met.
        while evals_left > 0
            # Step 2. Apply optimizer to cheap model
            candidate_solutions = solve(solver, cheaper_model)

            # Step 3. Evaluate solutions from 2, using the expensive model
            # Guarantee that the number of Max Evals is satisfied
            candidate_solutions, evals_left = clip(candidate_solutions, evals_left)
            solutions = evaluate(expensiv_model, candidate_solutions)

            # Step 4. Add results from 3 to ParetoResult
            push!(meta_solver, solutions)
            # Step 5. Update the surrogates
            foreach(correct,  objectives(meta_model))
        end
        # Step 7. Return Pareto Result non-dominated
        convert(Vector{Solution}, ParetoFront(s)...)
    end
