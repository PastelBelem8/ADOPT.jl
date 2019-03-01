# TODO
# 2. Case: 2+ points, transform the 2 points
# 3. Case: 2+ points + transform each point, filter

# -----------------------------------------------------------------------
#  Statistics Utils
# -----------------------------------------------------------------------
# TODO - Indicators
# 1. COLLECT MEASURE OF DISTRIBUTION / Diversity
# 2. Average Spacing
# 3. Ratio OVNG
# 4.
collect_statistics(ss::Vector{Solution}) = begin end

# -----------------------------------------------------------------------
#  Clustering / Distribution Utils
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Filtering Functions
# -----------------------------------------------------------------------
is_nondominated(solutions::Vector{Solution}) = let
    V = hcat(map(objectives, solutions)...)
    [solutions[i] for i in length(solutions)
                if Pareto.is_nondominated(V[:, i], V[:,1:end .!=i])]
    end
is_nondominated() = (solutions) -> is_nondominated(solutions)

is_feasible(solutions::Vector{Solution}) = filter(isfeasible, solutions)
is_feasible() = (solutions) -> is_feasible(solutions)

is_acceptable_penalty(solutions::Vector{Solution}, threshold) =
    filter(s -> s ≤ threshold, solutions)
is_acceptable_penalty(threshold) = (solutions) -> is_acceptable_penalty(solutions, threshold)

# Remove solutions that are too close
is_too_close(solutions::Vector{Solution}) = begin
    
end

export is_nondominated, is_feasible, is_acceptable_penalty, is_too_close


# -----------------------------------------------------------------------
# Solution Convert Routines
# -----------------------------------------------------------------------
convert(::Type{Solution}, x, y) = Solution( convert(typeof_variables(Solution), x),
                                            convert(typeof_objectives(Solution), y))
convert(::Type{Solution}, x, y, constraints) =
    let variables = convert(typeof_variables(Solution), x)
        objectives = convert(typeof_objectives(Solution), y)

        # Constraints
        constraints = convert(typeof_constraints(Solution), map(c -> evaluate(c, x...), constrs))
        constraint_violation = convert(typeof_constraint_violation(Solution), evaluate_penalty(constrs, x...))

        # Booleans
        feasible = constraint_violation != 0
        evaluated = true

        Solution(variables, objectives, constraints, constraint_violation, feasible, evaluated)
    end

convert(::Type{Vector{Solution}}, X, y, constraints) =
    if isempty(constraints)
        map(1:size(X, 2)) do sample
            convert(Solution, X[:, sample], y[:, sample]) end
    else
        map(1:size(X, 2)) do sample
            convert(Solution, X[:, sample], y[:, sample], constraints[:, sample]) end
    end

# -----------------------------------------------------------------------
# Sampling Solver
# -----------------------------------------------------------------------
struct SamplingSolver <: AbstractSolver
    algorithm_params::Dict{Symbol,Any}

    max_evaluations::Int
    filtering_function::Function

    nondominated_only::Bool

    SamplingSolver(;algorithm_params=Dict{Symbol, Any}(), max_evaluations=100,
                    filtering_f=(_...)->true, nondominated_only=true) =
        begin
            check_arguments(algorithm, algorithm_params, max_evaluations)
            new(algorithm, algorithm_params, max_evaluations, nondominated_only)
        end
end

# Arguments Validation
check_arguments(::Type{SamplingSolver}, algorithm, algorithm_params, max_evaluations) =
    if max_evaluations < 0
        throw(DomainError("invalid value of $max_evaluations for parameter `max_evaluations` must be positive"))
    end

# Selectors
algorithm(s::SamplingSolver) = s.algorithm
algorithm_params(s::SamplingSolver) = s.algorithm_params

max_evaluations(s::SamplingSolver) = s.max_evaluations
filtering_function(s::SamplingSolver, Xs) = s.filtering_function(Xs)

# Solver routines -------------------------------------------------------
solve(solver::SamplingSolver, model::Model) =
    let nvars = nvariables(model)
        nobjs = nobjectives(model)
        ncnstrs = nconstraints(model)
        unsclrs = unscalers(model)
        algorithm_params = algorithm_params(solver)
        function evaluation_f(x)
            sol = evaluate(model, x);
            hcat(objectives(sol), constraints(sol)...)
        end

        @info "[$(now())] Creating samples..."
        X, y = create_samples(; nvars=nvars, evaluate=evaluate_f, unscalers=unsclrs,
                                algorithm_params..., nsamples=max_evaluations(solver))
        y_objs, y_constrs = size(y, 1) == nobjs ? (y, Real[]) : (y[1:nobjs], y[nobjs:end])

        @info "[$(now())] Successfully loaded $(size(X, 2)) samples..."
        solutions = convert(Vector{Solution}, X, y_objs, y_constrs)

        @info "[$(now())] Collecting statistical measurements before filtering..."
        solutions_stats = collect_statistics(solutions)

        @info "[$(now())] Filtering solutions..."
        final_solutions = filtering_function(solver, solutions)

        @info "[$(now())] Collecting statistical measurements to filtered solutions..."
        final_solutions_stats = collect_statistics(solutions)

        return final_solutions
    end
