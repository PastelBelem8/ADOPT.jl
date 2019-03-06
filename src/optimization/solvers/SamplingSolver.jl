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
    feasible_filter::Function

    max_evaluations::Int
    nondominated_only::Bool

    SamplingSolver(;algorithm_params, max_eval=100, constraint_type=:hard, threshold=0.01, nondominated_only=true) = begin
        check_arguments(algorithm_params, max_eval, constraint_type, threshold)

        feasible_filter = constraint_type == :hard ? hard_constraints : soft_constraints(threshold)
        new(algorithm_params, feasible_filter, max_eval, nondominated_only)
    end
end

# Arguments Validation
check_arguments(::Type{SamplingSolver}, algorithm_params, max_eval, constraint_type, threshold) =
    if max_eval < 0
        throw(DomainError("invalid value of $max_eval for parameter `max_eval` must be positive"))
    elseif ! constraint_type in (:hard, :soft)
        throw(DomainError("invalid value of $constraint_type for parameter `constraint_type` must be :hard or :soft"))
    elseif constraint_type == :soft && threshold < 0
        throw(DomainError("invalid value of $threshold for parameter `threshold` must be positive"))
    end

# Selectors
algorithm(s::SamplingSolver) = s.algorithm
algorithm_param(s::SamplingSolver, param::String) =
    get(s.algorithm_params, param, nothing)
algorithm_params(s::SamplingSolver) = s.algorithm_params
max_evaluations(s::SamplingSolver) = s.max_evaluations

"Returns the solutions in ss that are feasible"
get_feasibles(solver::SamplingSolver, ss::Vector{Solution}) = solver.feasible_filter(ss)

# Predicates
nondominated_only(solver::SamplingSolver) =  solver.nondominated_only


solve_it(solver::SamplingSolver, model::Model) = let
    nvars = nvariables(model)
    nobjs = nobjectives(model)
    unsclrs = unscalers(model)
    a_params = algorithm_params(solver)
    function evaluation_f(x)
        sol = evaluate(model, x);
        vcat(objectives(sol), constraints(sol))
    end

    @debug "[$(now())][SamplingSolver] Creating samples..."
    X, y = create_samples(; nvars=nvars, evaluate=evaluation_f, unscalers=unsclrs,
                            a_params..., nsamples=max_evaluations(solver))
    y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs,:], y[nobjs+1:end,:])

    @debug "[$(now())][SamplingSolver] Successfully evaluated $(size(X, 2)) samples..."
    solutions = convert(Vector{Solution}, X, y_objs, constraints(model), y_constrs)

    @debug "[$(now())][SamplingSolver] Removing infeasible solutions..."
    solutions = get_feasibles(solver, solutions)

    nondominated_only(solver) ? Pareto.is_nondominated(solutions) : solutions
end

get_solver(::Type{SamplingSolver}, algorithm, params, evals, nd_only) = let
    solver_params = filter(p -> p[1] ∈ (:threshold, :constraint_type), params)
    solver_params = Dict(solver_params)
    SamplingSolver(;algorithm_params=merge(params, Dict(:sampling_function => algorithm)),
                    max_eval=evals, nondominated_only=nd_only, solver_params...)
    end

# Create strategies
# 1. Sample until no more evals left
# 2. Sample but restrict the bounds (care to not fix any variable...) - keep it simple
#       Problems - how to decide which one is the best ?
#       Split evals left amongst best solutions?


# Strategy 1. Normal the same as solve_it but wrap it in a while loop

solve_it(solver::SamplingSolver, model::Model) = let
    nvars = nvariables(model)
    nobjs = nobjectives(model)
    unsclrs = unscalers(model)
    a_params = algorithm_params(solver)
    function evaluation_f(x)
        sol = evaluate(model, x);
        vcat(objectives(sol), constraints(sol))
    end

    @debug "[$(now())][SamplingSolver] Creating samples..."
    X, y = create_samples(; nvars=nvars, evaluate=evaluation_f, unscalers=unsclrs,
                            a_params..., nsamples=max_evaluations(solver))
    y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs,:], y[nobjs+1:end,:])

    @debug "[$(now())][SamplingSolver] Successfully evaluated $(size(X, 2)) samples..."
    solutions = convert(Vector{Solution}, X, y_objs, constraints(model), y_constrs)

    @debug "[$(now())][SamplingSolver] Removing infeasible solutions..."
    solutions = get_feasibles(solver, solutions)

    nondominated_only(solver) ? Pareto.is_nondominated(solutions) : solutions
end


run_iteration(nvars::Int, nobjs::Int, params, filter::Function, model::Model) = let
    unsclrs = unscalers(model)
    evaluation_f(x) = let   s = evaluate(model, x);
                            vcat(objectives(s), constraints(s))
                        end
    @debug "[$(now())][SamplingSolver][run_iteration] Creating samples..."
    X, y = create_samples(; nvars=nvars, evaluate=evaluation_f, unscalers=unsclrs,
                            params..., nsamples=evals_left)
    y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs,:], y[nobjs+1:end,:])

    @debug "[$(now())][SamplingSolver][run_iteration] Successfully evaluated $(size(X, 2)) samples..."
    candidate_solutions = convert(Vector{Solution}, X, y_objs, constraints(model), y_constrs)

    @debug "[$(now())][SamplingSolver][run_iteration] Filtering solutions using $(string(foo))..."
    filter(candidate_solutions), length(candidate_solutions)
end

normal_case(solver, model) =  let
    nvars, nobjs = nvariables(model), nobjectives(model);
    a_params = algorithm_params(solver)
    filter = (solutions) -> get_feasibles(solver, solutions)

    solutions, _ = run_iteration(nvars, nobjs, a_params, filter, model)
    nondominated_only(solver) ? Pareto.is_nondominated(solutions) : solutions
end

iterative_case(solver, model) = let
    solutions = []
    a_params = algorithm_params(solver)
    evals_left = max_evaluations(solver);
    nvars, nobjs = nvariables(model), nobjectives(model);
    filter = (solutions) -> get_feasibles(solver, solutions)

    while evals_left > 0
        candidate_solutions, evals = run_iteration(nvars, nobjs, a_params, filter, model)
        push!(solutions, candidate_solutions)
        evals_left -= evals
    end
    nondominated_only(solver) ? Pareto.is_nondominated(solutions) : solutions
end


focused_case(solver, model) = let

end
