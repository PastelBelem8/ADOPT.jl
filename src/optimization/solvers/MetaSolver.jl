using .Metamodels
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
sample(;nsamples, sampling, unscalers, evaluate) =
    let nvars = length(unscalers)
        X = sampling(nvars, nsamples)
        foreach((unscale, var) -> X[var,:] = unscale(X[var,:], 0, 1), unscalers, 1:nvars)
        y = mapslices(evaluate, X, dims=1)
        X, y
    end

sample_to_file(;nsamples, sampling, unscalers, evaluate, filename, header=nothing, dlm=',', _...) =
    let (X, y) = sample(nsamples=nsamples, sampling=sampling, unscalers=unscalers, evaluate=evaluate)
        data = vcat(X, y)'
        open(filename, "w") do io
            if header != nothing
                join(io, header, dlm)
                write(io, '\n')
            end
            writedlm(io, data, dlm)
        end
        X, y
    end


from_file(;nvars::Int=0, vars_cols, objs_cols, filename, has_header::Bool=true, dlm=',', _...) =
    let data = open(filename, "r") do io
                    has_header ? readline(io) : nothing;
                    readdlm(io, dlm, Float64, '\n')
                end
        X, y = data[:, vars_cols], data[:, (nvars .+ objs_cols)]
        X', y'
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
    meta_model::Any
    objectives::Tuple{Vararg{AbstractObjective}}

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

    Surrogate(meta_model; objectives::Tuple{Vararg{AbstractObjective}}, creation_f::Function=Metamodels.fit!,
              creation_params::Dict{Symbol, Any}, correction_f::Function=Metamodels.fit!,
              correction_frequency::Int=1, evaluation_f::Function=Metamodels.predict,
              decay_rate::Real=0.1) = begin
        if isempty(objectives)
            throw(DomainError("invalid argument `objectives` cannot be empty."))
        elseif correction_frequency < 0
            throw(DomainError("invalid argument `correction_frequency` must be a non-negative integer."))
        elseif decay_rate < 0
            throw(DomainError("invalid argument `decay_rate` must be a non-negative real."))
        end
        new(meta_model, objectives, creation_f, creation_params, correction_f, correction_frequency, evaluation_f, decay_rate)
    end
end

# Selectors
@inline surrogate(s::Surrogate) = s.meta_model
@inline objectives(s::Surrogate) = s.objectives

@inline nobjectives(s::Surrogate) = sum(map(nobjectives, objectives(s)))

@inline correction_function(s::Surrogate) = s.correction
@inline correction_function(s::Surrogate, X, y) = s.correction(surrogate(s), X, y)

@inline creation_function(s::Surrogate) = s.creation
@inline creation_function(s::Surrogate, X, y) = s.creation(surrogate(s), X, y)

@inline evaluation_function(s::Surrogate) = s.evaluation
@inline evaluation_function(s::Surrogate, X) = s.evaluation(s.meta_model, X)

@inline creation_params(s::Surrogate) = s.creation_params
@inline creation_param(s::Surrogate, param::Symbol, default) = get(creation_params(s), param, default)

# Predicate
@inline is_multi_target(s::Surrogate) = nobjectives(s) > 1

# Modifiers
create!(surrogate::Surrogate, unscalers) =
    let λ = creation_param(surrogate, :sampling_function, nothing)
        λparams = creation_params(surrogate)

        # Sampling
        if λ != nothing
            sampling = Sampling.exists(λ) ? Sampling.get_existing(λ; λparams...) : λ

            evaluate(x...) = map(o -> apply(o, x...), objectives(surrogate)) |> flatten
            filename = creation_param(surrogate, :filename, "sample-$(string(λ))-$(now()).csv")

            X, y = sample_to_file(; λparams..., sampling=λ, evaluate=evaluate,
                                    unscalers=unscalers, filename=filename)
        # From File
        else
            X, y = from_file(;λparams...)
        end
        @info "[$(now())] Training surrogate model for $(size(X, 2)) samples with $(size(X, 1)) dimensions";
        creation_function(surrogate, X, y);
    end

correct!(surrogate::Surrogate, data::Vector{Solution}) =
    let nsols = length(data)
        X = hcat(map(variables, data)...)
        y = hcat(map(objectives, data)...)
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
    pareto_result::ParetoResult

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

"Returns the Pareto Front solutions obtained with the MetaSolver"
ParetoFront(s::MetaSolver) = Pareto.ParetoFront(results(s))

# Modifiers
"Stores the variables and objectives in the corresponding meta solver"
Base.push!(solver::MetaSolver, solutions::Vector{Solution}) =
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
        unsclrs = unscalers(meta_model)
        evals_left = max_evaluations(meta_solver)
        create(s) = create!(s, unsclrs)
        correct(solutions) = surrogate ->
                                let err = correct!(surrogate, solutions)
                                    @info "[$(now())] Retrained surrogate exhibits $(err)% error."
                                    err
                                end

        # Models
        cheaper_model = cheap_model(meta_model, dynamic=true)
        expensiv_model = expensive_model(meta_model)

        # Step 1. Create each Surrogate
        foreach(create, unsafe_surrogates(meta_model))

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
            foreach(correct(solutions),  unsafe_surrogates(meta_model))
        end
        # Step 7. Return Pareto Result non-dominated
        convert(Vector{Solution}, ParetoFront(meta_solver)..., constraints(expensiv_model))
    end

export MetaSolver, solve
