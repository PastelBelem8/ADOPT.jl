# TODO
# 2. Case: 2+ points, transform the 2 points
# 3. Case: 2+ points + transform each point, filter
using .Sampling

# -----------------------------------------------------------------------
# Filtering Functions
# -----------------------------------------------------------------------
hard_constraints(solutions::Vector{Solution}) = filter(isfeasible, solutions)

soft_constraints(solutions::Vector{Solution}, threshold) =
    filter(s -> constraint_violation(s) ≤ threshold, solutions)
soft_constraints(threshold) = (solutions) -> soft_constraints(solutions, threshold)

# -----------------------------------------------------------------------
# Sampling Solver
# -----------------------------------------------------------------------
struct SamplingSolver <: AbstractSolver
    algorithm_params::Dict{Symbol,Any}
    constraint_type::Function
    sampling_strategy::Function

    max_evaluations::Int
    nondominated_only::Bool

    SamplingSolver(;algorithm_params, max_eval=100, sampling_strategy=:simple, constraint_type=:hard, threshold=0.01, nondominated_only=true) = begin
        println(sampling_strategy)
        check_arguments(algorithm_params, max_eval, sampling_strategy, constraint_type, threshold)
        sampling_strategy = sampling_strategy == :simple ? simple_strategy : iterative_strategy
        constraint_type = constraint_type == :hard ? hard_constraints : soft_constraints(threshold)
        new(algorithm_params, constraint_type, sampling_strategy, max_eval, nondominated_only)
    end
end

# Arguments Validation
check_arguments(::Type{SamplingSolver}, algorithm_params, max_eval, sampling_strategy, constraint_type, threshold) =
    if max_eval < 0
        throw(DomainError("invalid value of $max_eval for parameter `max_eval` must be positive"))
    elseif ! constraint_type in (:hard, :soft)
        throw(DomainError("invalid value of $constraint_type for parameter `constraint_type` must be :hard or :soft"))
    elseif constraint_type == :soft && threshold < 0
        throw(DomainError("invalid value of $threshold for parameter `threshold` must be positive"))
    elseif !sampling_strategy in (:simple, :iterative)
        throw(DomainError("invalid value of $sampling_strategy for parameter `sampling_strategy` must be :simple or :iterative"))
    end

# Selectors
algorithm(s::SamplingSolver) = s.algorithm
algorithm_param(s::SamplingSolver, param::String) =
    get(s.algorithm_params, param, nothing)
algorithm_params(s::SamplingSolver) = s.algorithm_params
max_evaluations(s::SamplingSolver) = s.max_evaluations

sampling_strategy(s::SamplingSolver) = s.sampling_strategy

"Returns the solutions in ss that are feasible"
get_feasibles(solver::SamplingSolver, ss::Vector{Solution}) = solver.constraint_type(ss)

# Predicates
nondominated_only(solver::SamplingSolver) =  solver.nondominated_only

# Solve routines -----------------------------------------------------
get_solver(::Type{SamplingSolver}, algorithm, params, evals, nd_only) = let
    solver_params = filter(p -> p[1] ∈ (:threshold, :constraint_type, :sampling_strategy), params)
    solver_params = Dict(solver_params)
    SamplingSolver(;algorithm_params=merge(params, Dict(:sampling_function => algorithm)),
                    max_eval=evals, nondominated_only=nd_only,
                    solver_params...)
    end

run_iteration(nvars::Int, nobjs::Int, nsamples, params, filter::Function, model::Model) = let
    unsclrs = unscalers(model)
    evaluation_f(x) = let   s = evaluate(model, x);
                            vcat(objectives(s), constraints(s))
                        end
    @debug "[$(now())][SamplingSolver][run_iteration] Creating samples..."
    X, y = create_samples(; nvars=nvars, evaluate=evaluation_f, unscalers=unsclrs,
                            params..., nsamples=nsamples)
    y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs,:], y[nobjs+1:end,:])

    @debug "[$(now())][SamplingSolver][run_iteration] Successfully evaluated $(size(X, 2)) samples..."
    candidate_solutions = convert(Vector{Solution}, X, y_objs, constraints(model), y_constrs)

    @debug "[$(now())][SamplingSolver][run_iteration] Filtering solutions using $(string(foo))..."
    filter(candidate_solutions), length(candidate_solutions)
end

simple_strategy(solver, model) =  let
    nvars, nobjs = nvariables(model), nobjectives(model);
    a_params = algorithm_params(solver)
    filter = (solutions) -> get_feasibles(solver, solutions)
    nsamples = get(a_params, :nsamples, max_evaluations(solver))

    solutions, _ = run_iteration(nvars, nobjs, nsamples, a_params, filter, model)
    solutions
end

iterative_strategy(solver, model) = let
    solutions = Solution[]
    a_params = algorithm_params(solver)
    evals_left = max_evaluations(solver);
    nvars, nobjs = nvariables(model), nobjectives(model);
    filter = (solutions) -> get_feasibles(solver, solutions)
    nsamples = get(a_params, :nsamples, evals_left)

    while evals_left > 0
        @debug "[$(now())][SamplingSolver][iterative_strategy] Evaluations left: $(evals_left)..."
        nsamples = min(nsamples, evals_left)
        candidate_solutions, evals = run_iteration(nvars, nobjs, nsamples, a_params, filter, model)
        foreach(candidate -> push!(solutions, candidate), candidate_solutions)
        evals_left -= evals
    end
    solutions
end

solve_it(solver::SamplingSolver, model::Model) =
    sampling_strategy(solver)(solver, model) |>
    (sols) ->  (nondominated_only(solver) ? Pareto.is_nondominated(sols) : sols)
