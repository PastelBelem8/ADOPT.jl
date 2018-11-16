using .SurrogateModels

struct SurrogateSolver <: AbstractSolver
    algorithm::Type
    algorithm_params::Dict{Symbol, Any}
    max_evaluations::Int

    correction_function::Function
    evaluator_function::Function
    sampling_function::Function
    validation_function::Function

    function SurrogateSolver(algorithm; algorithm_params, max_eval,
                            correction_f, evaluator_f, sampling_f, validate_f)
        #TODO - Add Check Arguments routine
        new(algorithm, algorithm_params, max_ev, correction_f, evaluator_f,
            sampling_f, validate_f)
    end
end

# Selectors
get_algorithm(solver::SurrogateSolver) = solver.algorithm
get_algorithm_params(solver::SurrogateSolver) = solver.algorithm_params
get_algorithm_param(solver::SurrogateSolver, param::Symbol, default=nothing) =
    get(get_algorithm_params(solver), param, default)
get_max_evaluations(solver::SurrogateSolver) = solver.max_evaluations

get_correction_function(solver::SurrogateSolver) = solver.correction_function
get_evaluator_function(solver::SurrogateSolver)  = solver.evaluator_function
get_sampling_function(solver::SurrogateSolver)   = solver.sampling_function
get_validation_function(solver::SurrogateSolver) = solver.validation_function

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
"""
    create_initial_surrogate()

Generates an initial surrogate model that is approximated using the `algorithm`
and the parameters specified in the solver.

The initial surrogate model can be created using sampling (or design of
experiments (DOE)) techniques, or by loading the data from csv files.
The csv files should only contain the decision variables' values and the
objectives' values.

# Arguments

"""
function create_surrogate(algorithm, X, y)
    # Cross-validation
    surrogate = fit!(algorithm, X, y)
    # TODO - Validate
    # TODO - If performance <= 0.6
    #           retrain (different params, more samples)
end


"Reads the necessary data to create the initial surrogate model from the file `filename`."
function create_initial_surrogate(algorithm, model::Model; filename::String)
    nvars = nvariables(model)
    # Read data from file
    data = map(x -> parse(Float64, x), csv_read(filename))
    X, y = data[:, 1:nvars], data[:, nvars:end]

    create_surrogate(algorithm, X, y)
end

"Uses a sampling function to generate the data to create the initial surrogate model with."
function create_initial_surrogate(algorithm, model::Model; nsamples::Int, sampling_f::Function)
    vars = variables(model)
    nvars, nobjs, nconstrs = length(vars), nobjectives(model), nconstraints(model)
    min_i(i) = lower_bound(vars[i])
    max_i(i) = upper_bound(vars[i])
    get_y(sol) = vcat(objectives(sol), constraints(sol))

    X = sampling_f(nvars, nsamples)' # Matrix is nsamples x nvars
    X = [map(v -> unscale(v, min_i(j), max_i(j)), X[:, j]) for j in 1:nvars]

    y = [get_y(evaluate(model, X[j, :])) for j in 1:nsamples]
    create_surrogate(algorithm, X, y)
end

function get_max_solutions(solutions, evals)
    if length(solutions) > evals
        solutions = solutions[1:evals]
    end
    evals -= length(solutions)
    solutions, evals
end

function new_substitute_model(model::Model, surrogate)
    original_vars = variables(model)
    original_objs = objectives(model)
    original_cnstrs = constraints(model)

    clone_and_replacef(obj, f) = Objective(f, coefficient(obj), sense(obj))

    objs = [clone_and_replacef(obj, x -> surrogate(x)[1]) for obj in original_objs]

end

function solve(solver::Solver, model::Model)
    corrector = get_correction_function(solver)
    evaluator = get_evaluator_function(solver)
    validator = get_validation_function(solver)

    # Create a copy of the model/problem, but replace Objective Functions
    surrogate = create_initial_surrogate(#TODO PASS PARAMETERS)
    surrogate_model = new_substitute_model(model, surrogate)

    evals = get_max_evaluations(solver)
    while evals > 0
        candidate_solutions = evaluator(surrogate_model)

        # Guarantee that the number of Max Evals is satisfied
        candidate_solutions, evals = get_max_solutions(candidate_solutions, evals)

        # For each candidate solution, evaluate its objectives
        sols = evaluate(model, candidate_solutions)
        surrogate = corrector(surrogate, sols)
        surrogate_error = validator(surrogate)

        @info "[$(now())] Retrained surrogate exhibits $(surrogate_error) % error."
        # TODO
        # Adaptively change the surrogate to give more weight to newly
        # obtained data, then to older one.
    end
end
