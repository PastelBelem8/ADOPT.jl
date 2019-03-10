# ---------------------------------------------------------------------
#                               Surrogate
# ---------------------------------------------------------------------
index_objectives(objectives) = let
    nobjs = map(nobjectives ∘ first, objectives)
    objs = map((o, nobj) -> length(o) == 2 ? o : (o[1], collect(1:nobj)), objectives, nobjs)
    # Compute Indexes and Offsets
    ix_offsets = foldl((a, b) -> push!(a, a[end]+b), nobjs, init=[0])[1:end-1]
    objs_ix = vcat(map((o, offset) -> o[2] .+ offset, objs, ix_offsets)...)

    collect(objs), collect(objs_ix)
    end
index_objectives(objectives::Vector{Union{T, Y}}) where{T<:AbstractObjective, Y<:AbstractObjective} =
    index_objectives(map(tuple, objectives))

index_objectives(objectives::Tuple{Vararg{Union{T, Y}}}) where{T<:AbstractObjective, Y<:AbstractObjective} =
    index_objectives(map(tuple, objectives))

"""
    Surrogate(type, (objective1, ..., objectiven))
    Surrogate(type, (objective1, ..., objectiven), f, f_params)
    Surrogate(type, (objective1, ..., objectiven), f, f_params, g, g_frequency)
    Surrogate(type, (objective1, ..., objectiven), f, f_params, g, g_frequency, η)

Represents an approximation (or meta) model of another function. Each meta model
possesses the list of objective functions it approximates, as well as the
creation and update methods they are built and updated with. The update
frequency is meant to limit the number of updates the meta model is to be
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

# Arguments
- `X::Any`: the data used for creating the approximation
- `y::Any`: the data used for creating the approximation
- `approximation_model::Any`: the model that creates the approximation based
on the data
- `objectives::Vector{Tuple{AbstractObjective, Vector{Int}}}`: a vector of
tuples with the objectives to approximate
- `objectives_indices::Vector{Int}`: the indices on the data that map to the
objectives to be approximated
- `variables_indices::Union{Colon, Vector{Int}}`: the indices on the data that
map to the variables to be used in the approximation (e.g., some objectives
might only use some variables)
- `creation::Function`: function that creates the initial approximation model
- `creation_params::Dict`: additional parameters to pass when creating the model
- `update::Function`: function that updates the approximation model
- `update_count::Int`: state variable to keep track of the updates
- `update_frequency::Int`: frequency of the updates
- `evaluation::Function`: function that makes prediction based on the
approximated model
"""
mutable struct Surrogate
    X::AbstractArray{Number,2}
    y::AbstractArray{Number,2}
    approximation_model::Any

    objectives::Vector{Tuple{AbstractObjective, Vector{Int}}}
    objectives_indices::Vector{Int}
    variables_indices::Union{Colon, Vector{Int}}

    # Create surrogate
    creation::Function
    creation_params::Dict{Symbol, Any}

    # Surrogates can be updated from t to t
    update::Function
    update_count::Real
    update_frequency::Real

    # Surrogates evaluate samples
    evaluation::Function

    Surrogate(approximation_model; objectives, variables_indices=(:),
              creation_f, creation_params=Dict{Symbol, Any}(), update_f=creation_f, update_frequency=1, evaluation_f) = begin
        check_arguments(Surrogate, approximation_model, objectives, variables_indices,
                        creation_f, creation_params, update_f,
                        update_frequency, evaluation_f, decay_rate)
        objectives, objectives_indices = index_objectives(objectives)
        new(Array{Int64,2}(undef, 0, 0), Array{Int64,2}(undef, 0, 0), approximation_model, objectives, objectives_indices, variables_indices,
            creation_f, creation_params, update_f, 0, 1/update_frequency, evaluation_f)
    end
end

# Arguments Validation
check_arguments(::Type{Surrogate}, Xs, ys, approximation_model, objs, var_ixs,
    creation_f, creation_params, update_f, update_frequency, evaluation_f) =
    if isempty(objs)
        throw(DomainError("invalid argument `objectives` cannot be empty."))
    elseif any(map(o -> isa(o, Tuple) && length(o) == 2 ? any(o[2] .< 1) : false, objs))
        throw(DomainError("invalid argument `objectives` should not non positive integers as indices."))
    elseif any(map(o -> isa(o, Tuple) && length(o) == 2 ? nonunique(o[2]) : false, objs))
        throw(DomainError("invalid argument `objectives` should not have repeated indexes."))
    elseif var_ixs != (:)
        if minimum(var_ixs) < 1
            throw(DomainError("invalid argument `var_ixs` cannot be smaller than 0."))
        elseif var_ixs != (:) && nonunique(var_ixs)
            throw(DomainError("invalid argument `var_ixs` cannot have repeated indexes."))
        end
    elseif !(0 < update_frequency <= 1)
        throw(DomainError("invalid argument `update_frequency` must in the range ]0,1]"))
    end

# Selectors
@inline sampling_transform(s::Surrogate) =
    (V) -> let  surrogate_vars = variable_indices(s)
                V_temp = zeros(maximum(surrogate_vars), size(V, 2))
                V_temp[surrogate_vars] = V
                V_temp
        end

@inline surrogate(s::Surrogate) = s.approximation_model
@inline objectives(s::Surrogate) = s.objectives
@inline nobjectives(s::Surrogate) =
    foldl((x ,y) -> x + length(y[2]), objectives(s), init=0)

@inline true_objectives(s::Surrogate) = map(first, objectives(s))
@inline coefficients(s::Surrogate) =
    let coeffs = foldl(vcat, map(coefficient, true_objectives(s)), init=Real[])
        coeffs[objectives_indices(s)]
    end
@inline senses(s::Surrogate) =
    let senses = foldl(vcat, map(sense, true_objectives(s)), init=Symbol[])
        senses[objectives_indices(s)]
    end

@inline objectives(s::Surrogate, y) = y[objectives_indices(s),:]
@inline variables(s::Surrogate, X) = X[variables_indices(s),:]

@inline objectives_indices(s::Surrogate) = s.objectives_indices
@inline variables_indices(s::Surrogate) = s.variables_indices

@inline creation_params(s::Surrogate) = s.creation_params

# Predicate
"Returns true if the surrogate interpolates more than one objective function"
@inline is_multi_target(s::Surrogate) = nobjectives(s) > 1
@inline is_to_update(s::Surrogate) = s.update_count ≤ 0

# Modifiers
@inline push_X!(s::Surrogate, data) = s.X = [s.X data]
@inline push_Y!(s::Surrogate, data) = s.y = [s.y data]
Base.push!(s::Surrogate, Xs, Ys) = let
    if isempty(Xs) || isempty(Ys)
        throw(DomainError("empty array cannot be added to surrogate data"))
    elseif size(Xs, 2) != size(Ys, 2)
        throw(DimensionMismatch("the number of samples in X $(size(Xs, 2)) does not match the number of samples in Y $(size(Ys, 2))"))
    end
    s.update_count -= size(Xs, 2)
    push_X!(s, variables(s, Xs))
    push_Y!(s, objectives(s, Ys))
end

"Stores the variables and objectives in the corresponding surrogate"
update!(s::Surrogate, solutions::Vector{Solution}) = begin
    push!(s, hcat(map(variables, solutions)...), hcat(map(objectives, solutions)...))
    if is_to_update(s)
        s.update(surrogate(s), variables(s, s.X), objectives(s, s.y))
        s.update_count = s.update_frequency
    end
end

create!(s::Surrogate, X, y) = begin
    s.X = X; s.y = y;
    s.creation(surrogate(s), X, y; creation_params(s)...)
end

@inline evaluate(s::Surrogate, X) =
    s.evaluation(s.approximation_model, variables(s, X))

export Surrogate

# ---------------------------------------------------------------------
#                               MetaSolver
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
    surrogates::Vector{Surrogate}
    sampling_params::Dict{Symbol, Any}

    max_evaluations::Int
    nondominated_only::Bool

    MetaSolver(solver; surrogates, sampling_params, max_eval=100, nondominated_only::Bool=true) = begin
        check_arguments(MetaSolver, surrogates, sampling_params, max_eval)
        new(solver, surrogates, sampling_params, max_eval, nondominated_only)
    end
end

# Arguments Validation
check_arguments(::Type{MetaSolver}, surrogates, nvars, nobjs, sampling_params, max_eval) =
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

# Selectors
@inline max_evaluations(s::MetaSolver) = s.max_evaluations
@inline optimiser(s::MetaSolver) = s.solver
@inline surrogates(s::MetaSolver) = s.surrogates

@inline nobjectives(s::MetaSolver) = sum(map(nobjectives, surrogates(s)))
@inline original_objectives(s::MetaSolver) = flatten(map(true_objectives, surrogates(s)))

@inline sampling_params(s::MetaSolver) = s.sampling_params
@inline sampling_data(s::MetaSolver) = (s.sampling_params[:X], s.sampling_params[:y])

@inline nondominated_only(s::MetaSolver) =  s.nondominated_only

# Predicates
sampling_required(s::MetaSolver) = !haskey(s.sampling_params, :X) || !haskey(s.sampling_params, :y)

# ---------------------------------------------------------------------
#                           Auxiliar Routines
# ---------------------------------------------------------------------
"Create a new cheaper model using the surrogates and replacing the expensive objective functions"
cheaper_model(surrogates::Vector{Surrogate}, model::Model) = let
    vars = variables(model)
    constrs = constraints(model)
    objs = map(surrogates) do surrogate
        λ = (x...) -> evaluate(surrogate, x...)
        coeffs = coefficients(surrogate)
        snses = senses(surrogate)

        # Create Objective
        is_multi_target(surrogate) ?
            SharedObjective(λ, coeffs, snses) :
            Objective(λ, coeffs..., snses...)
    end
    Model(vars, objs, constrs)
end

"Obtain the initial data from the model to create the surrogates with"
initial_data(meta_solver::MetaSolver, model::Model) = let
    nobjs = nobjectives(model)
    X, y = !sampling_required(meta_solver) ? sampling_data(meta_solver) : begin
        function evaluation_f(x)
            sol = evaluate(model, x);
            vcat(objectives(sol), constraints(sol))
        end
        create_samples(; nvars=nvariables(model), evaluate=evaluation_f,
                        unscalers=unscalers(model), sampling_params(meta_solver)...)
    end
    y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs,:], y[nobjs+1:end,:])
    X, y_objs, y_constrs
    end

solve_it(meta_solver::MetaSolver, model::Model) = let
    solver = optimiser(meta_solver)
    surrogatez = surrogates(meta_solver)
    cheap_model = cheaper_model(surrogatez, model)
    evals_left = max_evaluations(meta_solver)

    # Step 1. Obtain initial data (e.g., sample, read file)
    @debug "[$(now())][MetaSolver] Obtaining initial data..."
    X, y, cs = initial_data(meta_solver, model)

    results = convert(Vector{Solution}, X, y, constraints(model), cs) # In optimization
    evals_left -= length(results)

    # Step 2. Create initial surrogates
    @debug "[$(now())][MetaSolver] Creating initial Surrogates..."
    foreach(s -> create!(s, X, y), surrogatez)

    while evals_left > 0
        @debug "[$(now())][MetaSolver] $evals_left Expensive evaluations left"

        # Step 3. Apply Solver to optimize the surrogates
        @debug "[$(now())][MetaSolver] Optimizing surrogates..."
        candidate_solutions = with(results_file, "$(results_dir())/metamodels.dump") do
                                solve_it(solver, cheap_model) end
        ncandidates = length(candidate_solutions);

        # Step 4. Evaluate best solutions with the original model
        @debug "[$(now())][MetaSolver] Found $ncandidates candidate solutions..."
        solutions = evaluate(model, candidate_solutions)

        # Step 5. Update the surrogates
        if !isempty(solutions)
            @debug "[$(now())][MetaSolver] Updating surrogates with information about $(length(solutions)) solutions..."
            foreach((s) -> update!(s, solutions), surrogatez)

            # Update solutions
            foreach(solution -> push!(results, solution), solutions)
            evals_left -= ncandidates;
        end
    end

    nondominated_only(meta_solver) ? Pareto.is_nondominated(results) : results
end

export MetaSolver, solve
