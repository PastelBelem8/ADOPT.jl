using .Metamodels
using .Pareto: ParetoDominance

struct MetaSolver <: AbstractSolver
    surrogate::Type
    surrogate_params::Dict{Symbol, Any}
    # Solver
    solver::AbstractSolver

    # Surrogates can be loaded from files or by sampling
    surrogate_creation::Function
    surrogate_creation_params::Dict{Symbol, Any}

    # Surrogates can be updated from t to t
    surrogate_correction_frequency::Real
    surrogate_correction_function::Function

    # Surrogates should increase exploitation with the increase of evaluations
    surrogate_exploration_decay_rate::Real

    max_evaluations::Int
    pareto_data::ParetoDominance
    end

    # TODO - Constructor
end

# Selectors
init_function(s::MetaSolver) = s.surrogate_creation
init_function_params(s::MetaSolver) = s.surrogate_creation_params
max_evaluations(s::MetaSolver) = s.max_evaluations
pareto_data(s::MetaSolver) = s.pareto_data
surrogate_optimiser(s::MetaSolver) = s.solver

function data(s::MetaSolver)
    dt = pareto_data(s)
    variables(dt), objectives(dt) # X, y
end

using DelimitedFiles: readdlm, writedlm

struct Mode{x} end
DATA_LOAD = Mode{:DATA_LOAD}()
SAMPLES_LOAD = Mode{:SAMPLES_LOAD}()

function from_file(;mode::Mode{:DATA_LOAD}, evaluator::Function,
                   nvars::Int, filename::AbstractString)
    data = readdlm(filename, ',')
    data[:, 1:nvars]', data[:, nvars:end]'
end

function from_file(;mode::Mode{:SAMPLES_LOAD}, evaluator::Function,
                   nvars::Int, filename::AbstractString)
   get_targets(sol) = vcat(objectives(sol), constraints(sol))
    X = readdlm(filename, ',')'

    @debug "[$(now())] Evaluating $(size(X, 2)) samples..."
    y = mapslices(get_targets ∘ sample -> evaluator(sample), X, dims=1)

    open(filename, "w") do io
        writeldm(io, [X y], ',')
    end
    X, y
end

"Uses a sampling function to generate the data to create the initial surrogate model with."
function sample(;sampling_f::Function, evaluator::Function,
                nsamples::Int, model::Model, filename::AbstractString)
    get_targets(sol) = vcat(objectives(sol), constraints(sol))
    vars = variables(model); nvars = length(vars)

    X = sampling_f(nvars, nsamples) # nvars x nsamples
    X = [unscale(vars[i], X[i, :], 0, 1) for i in 1:nvars]

    @debug "[$(now())] Evaluating $nsamples samples..."
    y = mapslices(get_targets ∘ sample -> evaluator(sample), X, dims=1)

    open(filename, "w") do io
        writeldm(io, [X y], ',')
    end
    X, y
end

"Ensures the necessary `kwargs` are well specified for the `sample` function"
function sample_kwargs(;kwargs...)
    res = Dict{Symbol, Any}()
    res[:sampling_f] = get(kwargs, :sampling_f, throw(ArgumentError("sampling function must be specified")))
    res[:nsamples] = get(kwargs, :nsamples, 100)
    res[:filename] = get(kwargs, :filename, "sample-$(now()).csv")
    res
end

"Ensures the necessary `kwargs` are well specified for the `from_file` function"
function from_file_kwargs(;kwargs...)
    res = Dict{Symbol, Any}()
    res[:mode] = get(kwargs, :mode, throw(ArgumentError("mode must be specified")))
    res[:filename] = get(kwargs, :filename, throw(ArgumentError("filename must be specified")))
    res
end

function create_surrogate(model::Model, init_function::Function, init_kwargs...)
    kwargs = Dict{Symbol, Any}()
    if init_function == sample
        kwargs = sample_kwargs(init_kwargs...)
        kwargs[:model] = model
    elseif init_function == from_file
        kwargs = from_file_kwargs(init_kwargs...)
        kwargs[:nvars] = nvariables(model)
    else
        throw(ArgumentError("inexisting initialization routine cannot be used to create surrogate.
        Consider using `sample` or `from_file` routines."))
    end
    kwargs[:evaluator] = (x) -> evaluate(model, x)
    X, y = init_function(kwargs...)
    create_surrogate(algorithm, X, y)
end

# TODO - Refactor (will compute the same surrogate nobjective times)
create_proxy_model(proxy::Function, original_model::Model) =
    create_proxy_model(
        map(1:nobjectives(model)) do i
            i -> (x -> surrogate(x))[i]
        end,
        original_model)

create_proxy_model(proxies::Vector{Function}, original_model::Model)
    replace_obj_f(o, f) = Objective(f, coefficient(o), sense(o))

    new_objs = map(replace_obj_f, objectives(original_model), proxies)
    Model(variables(original_model), new_objs, constraints(original_model))
end

function clip(elements, nmax)
    if length(elements) > nmax
        elements = elements[1:nmax]
    end
    nmax -= length(elements)
    elements, nmax
end

function update_data!(solver::MetaSolver, new_data)
    vars = map(variables, new_data)
    objs = map(objectives, new_data)
    push!(solver.pareto_sets, vars, objs);
end

function update_surrogate!(surrogate, data)
    X, y = data
    # Fit surrogate
    fit!(surrogate, X, y)

    # TODO - Get error measurement
end

function solve(solver::MetaSolver, original_model::Model)
    data() = data(solver)
    expensive_solver = surrogate_optimiser(solver)
    surrogate = create_surrogate(original_model,
                                 init_function(solver),              # Initalization routine
                                 init_function_params(solver)...)
    proxy_model = create_proxy_model(surrogate, original_model)

    evals_left = max_evaluations(solver)
    while evals_left > 0
        # Obtain candidate solutions
        candidate_solutions = expensive_solver(expensive_solver, proxy_model)
        # Guarantee that the number of Max Evals is satisfied
        candidate_solutions, evals_left = clip(candidate_solutions, evals_left)

        # For each candidate solution, evaluate its objectives
        sols = evaluate(original_model, candidate_solutions)

        update_data!(solver, sols)
        surrogate, surrogate_error = update_surrogate!(surrogate, data())
        # TODO - Adaptively change the surrogate to give more weight to newly
        # obtained data, then to older one.
        @info "[$(now())] Retrained surrogate exhibits $(surrogate_error) % error."
    end

end

export solve, from_file, sample, DATA_LOAD, SAMPLES_LOAD, MetaSolver
