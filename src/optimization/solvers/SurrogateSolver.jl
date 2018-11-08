
"""
    SurrogateSolver

To solve a problem using the SurrogateSolver it is necessary to first model the
problem to be solved using the data structures provided by the module
[`Optimization`](@ref). After definin the model, i.e., specifying the variables,
the objectives, and the constraints if any. The user should then specify which
of the algorithms available in the [`Surrogate`](@ref) module, he would like
to use, as well as any parameters that are deemed mandatory or that the user
wishes to change.

The user should specify the basic routines to perform a normal surrogate-based
workflow as will explained in the following section.

# Arguments
- `algorithm::Type`: The algorithm type to create the surrogate with.
- `algorithm_params::Dict{Symbol, Any}`: Additional parameters to be passed to the
surrogate model that is going to be created by the `algorithm`.
- `max_evaluations::Int`: Number of maximum expensive evaluations.

- `sampling_function::Function`: Function responsible for the initial sampling
of data.
- `strategy_function::Function`: Function that explores the surrogate model
and selects infill points to be evaluated with the expensive function.
- `validation_function::Function`: Function that validates the prediction
capability of the model.
- `correction_function::Function`: Function that is responsible for correcting
the model.

"""
struct SurrogateSolver <: AbstractSolver
    algorithm::Type
    algorithm_params::Dict{Symbol, Any}
    max_evaluations::Int

    sampling_function::Function
    strategy_function::Function
    validation_function::Function
    correction_function::Function

    function SurrogateSolver(algorithm; algorithm_params, max_eval, sampling_f,
                                    strategy_f, validation_f, correction_f)
        check_arguments(SurrogateSolver, algorithm, algorithm_params, max_eval,
                        sampling_f, strategy_f, validation_f, correction_f)
        new(algorithm, algorithm_params, max_ev)
    end
end

function check_arguments(solver::Type{SurrogateSolver}, algorithm, algorithm_params,
                        max_eval, sampling_f, strategy_f, validation_f, correction_f)
    # TODO
end

get_algorithm(solver::SurrogateSolver) = solver.algorithm
get_algorithm_params(solver::SurrogateSolver) = solver.algorithm_params
get_algorithm_param(solver::SurrogateSolver, param::Symbol, default=nothing) =
    get(get_algorithm_params(solver), param, default)
get_max_evaluations(solver::SurrogateSolver) = solver.max_evaluations

get_sampling_function(solver::SurrogateSolver) = solver.sampling_function
get_strategy_function(solver::SurrogateSolver) = solver.strategy_function
get_validation_function(solver::SurrogateSolver) = solver.validation_function
get_correction_function(solver::SurrogateSolver) = solver.correction_function


"""
  check_params(solver, model)

"""
function check_params(solver::SurrogateSolver, model::Model)

end

# Solver Routines
function solve(solver::SurrogateSolver, model::Model)
    check_params(solver, model)

    # Step 1. Generate initial surrogate model
    surrogate = generate_surrogate(solver, model)

    remaining_evals = get_max_evaluations(solver)
    while remaining_evals > 0
        # Step 2. Obtain approximate solution by optimizing the surrogate
        approximate_sols = solve(surrogate, model)

        # Step 3. Use the high-fidelity (original) model to evaluate
        approximate_sols, remaining_evals = get_remaining(approximate_sols, max_eval)
        original_sols = evaluate(solver, model, approximate_sols)

        # Step 4. Correct the surrogate model using the newly obtained data
        surrogate = update_surrogate(solver, surrogate, original_sols)
    end
    # TODO - What to return here?
end

"Generate the initial surrogate using multiple "
function generate_surrogate(solver::SurrogateSolver, model::Model)
    ndims = nvariables(model)
    nsamples = get_algorithm_param(solver, :nsamples, -1)
    sampling = get_sampling_function(solver)

    # 1. Apply Sampling Function
    # 2. Evaluate the Samples
    # 3. Create the Surrogate provide a measure of performance/error
end

#FIXME - Fix the name
function get_remaining(approximate_sols, total)
    remaining = length(approximate_sols)
    nremaining = total - remaining
    if 0 <= nremaining
        approximate_sols, nremaining
    else
        approximate_sols[1:total], 0
    end
end

function evaluate(solver, model, approximate_sols)
    # Step 1. Use original high-fidelity models (encoded in Objectives)
    # Step 2. Evaluate each approximate_sol
end


function update_surrogate(solver, surrogate, original_sols)
    # Step 1. Update the Surrogate
    # FIXME - Understand if the user is the responsible for specifying the
    # correction or if it is up to the own surrogate.
    # Step 2. Compute the
end
