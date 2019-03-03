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
algorithm_params(s::SamplingSolver) = s.algorithm_params
max_evaluations(s::SamplingSolver) = s.max_evaluations

"Returns the solutions in ss that are feasible"
get_feasibles(solver::SamplingSolver, ss::Vector{Solution}) = solver.feasible_filter(ss)

# Predicates
nondominated_only(solver::SamplingSolver) =  solver.nondominated_only


# Solver routines -------------------------------------------------------
solve(solver::SamplingSolver, model::Model) = let
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
