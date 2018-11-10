# Using
using .Platypus

# Imports
import Base: convert

"Converts a dictionary to an array of pairs to be passed as kwargs"
dict2expr(dict::Dict) = [kv for kv in dict]

# Converter routines ----------------------------------------------------
# These routines provide the interface between the solver and the
# Platypus Python library.
# -----------------------------------------------------------------------
function platypus_function(objectives, constraints)::Function
    length(constraints) > 0  ? ((x...) -> return [func(o)(x...) for o in objectives],
                                                 [func(c)(x...) for c in constraints]) :
                               ((x...) -> return [func(o)(x...) for o in objectives])
end

function platypus_function_with_profiling(objectives, constraints)::Function

  function aggregate_function(x...)
    local profiling_times = Vector{Real}()
    local profiling_results = Vector{Real}()

    function profile(f, args...)
      st = time();
      res = f(args...)
      @info "[$(now())] Objective function result (args: $(args...)): $res"
      push!(profiling_results, res)
      push!(profiling_times, time() - st)
      res
    end
    @info "[$(now())] Running aggregate objective function for parameters: $(x...)"
    prepend!(profiling_results, x...) # Decision variables
    start_time = time()

    results = [profile(func(o), x...) for o in objectives]

    if length(constraints) > 0
      results = results, [profile(func(c), x...) for c in constraints]
    end
    prepend!(profiling_times, time() - start_time)

    # Write to File
    output = vcat(profiling_times, profiling_results)
    csv_write("results.csv", output)
    # Return Results
    return results
  end

  return (x...) -> aggregate_function(x...)
end
# ------------------------------------------------------------------------
# Problem Converter Routines
# ------------------------------------------------------------------------
function convert(::Type{Platypus.Problem}, m::Model)
  # 1. Create Base Problem
  nvars, nobjs, nconstrs = nvariables(m), nobjectives(m), nconstraints(m)
  problem = Platypus.Problem(nvars, nobjs, nconstrs)

  # 2. Refine the Problem instance
  # 2.1. Convert types
  # TODO - fix PLATYPUS_WRAPPER TO be something more specific - e.g. PlatypusType
  Platypus.set_types!(problem, [convert(Platypus.PlatypusWrapper, v)
                                        for v in variables(m)])

  # 2.2. Convert Constraints
  constrs = []
  if nconstrs > 0
    constrs = constraints(m)
    Platypus.set_constraints!(problem, [convert(Platypus.Constraint, c)
                                                      for c in constrs])
  end

  # 2.3. Convert Objective Function
  objs = objectives(m)
  Platypus.set_directions!(problem, directions(objs))
  Platypus.set_function!(problem, platypus_function_with_profiling(objs, constrs))
  problem
end
convert(::Type{Platypus.Constraint}, c::Constraint) = "$(operator(c))0"

# ------------------------------------------------------------------------
# Variable Converter Routines
# ------------------------------------------------------------------------
convert(::Type{Platypus.PlatypusWrapper}, variable::IntVariable) =
    Platypus.Integer(lower_bound(variable), upper_bound(variable))
convert(::Type{Platypus.PlatypusWrapper}, variable::RealVariable) =
    Platypus.Real(lower_bound(variable), upper_bound(variable))

# ------------------------------------------------------------------------
# Solution Converter Routines
# ------------------------------------------------------------------------
function convert(::Type{Solution}, s::Platypus.Solution)
  variables = convert(typeof_variables(Solution), Platypus.get_variables(s))
  objectives = convert(typeof_objectives(Solution), Platypus.get_objectives(s))
  constraints = convert(typeof_constraints(Solution), Platypus.get_constraints(s))
  constraint_violation = convert(typeof_constraint_violation(Solution), Platypus.get_constraint_violation(s))
  feasible = convert(typeof_feasible(Solution), Platypus.get_feasible(s))
  evaluated = convert(typeof_evaluated(Solution),Platypus.get_evaluated(s))

  Solution(variables, objectives, constraints, constraint_violation, feasible, evaluated)
end
convert(::Type{Vector{Solution}}, ss::Vector{Platypus.Solution}) =
    [convert(Solution, s) for s in ss]

# ------------------------------------------------------------------------
# Generators Converter Routines
# ------------------------------------------------------------------------
convert(::Type{Platypus.RandomGenerator}, ::Dict{Symbol, T}) where{T} =
    Platypus.RandomGenerator()

# ------------------------------------------------------------------------
# Selectors Converter Routines
# ------------------------------------------------------------------------
# FIXME - There is no support to other dominance objects
convert(::Type{Platypus.TournamentSelector}, params::Dict{Symbol, T}) where{T} =
  Platypus.TournamentSelector(:($(dict2expr(params)...)))

# ------------------------------------------------------------------------
# Variators Converter Routines
# ------------------------------------------------------------------------
function convert(::Type{Platypus.SimpleVariator}, params::Dict{Symbol, T}) where{T}
  variator, variator_args = params[:name], filter(p -> first(p) != :name, params)
  mandatory_params_satisfied(variator, variator_args)
  variator(:($(dict2expr(variator_args))...))
end

"""
# Example

julia> variator = Dict(:name=>SBX, :probability=>0.5, :distribution_index=>12)
Dict{Symbol,Any} with 3 entries:
  :probability        => 0.5
  :distribution_index => 12
  :name               => SBX

julia> convert(Platypus.SimpleVariator, variator)
<platypus.operators.SBX object at 0x00000000027785F8>
"""
function convert(::Type{Platypus.CompoundVariator}, params::Dict{Symbol, T}) where{T}
  variator, variator_args = params[:name], filter(p -> first(p) != :name, params)
  mandatory_params_satisfied(variator, variator_args)
  # If it is a dictionary, then it is a variator, else is a simple value
  mk_conv(v) = (v) -> isa(v, Dict) ? convert(supertype(v[:name]), v) : v

  for (k, v) in variator_args
    args[k] = [mkconv(vi) for vi in v]
  end
  variator(:($(dict2expr(args)...)))
end

function convert_params(params::Dict{Symbol, T}) where{T}
  if (generator_params = get(params, :generator, nothing)) != nothing
    params[:generator] = convert(generator_params[:name], generator_params)
  end
  if (variator_params = get(params, :variator, nothing)) != nothing
    params[:variator] = convert(supertype(variator_params[:name]), variator_params)
  end
  if (selector_params = get(params, :selector, nothing)) != nothing
    params[:selector] = convert(selector_params[:name], selector_params)
  end

  params
end

# ------------------------------------------------------------------------
# Solver Routines
# ------------------------------------------------------------------------
"""
    PlatypusSolver

To solve a problem using the PlatypusSolver it is necessary to first model the
problem to be solved using the data structures provided by the module
`Optimization`. After defining the model, i.e., specifying the variables,
the objectives, and the constraints if any, the user should then specify
which of the algorithms available in Platypus, he would like to use, as well as
any parameters that are deemed mandatory or that the user wishes to change.

By convention, the more complex parameters such as `Variator`, `Generator`, and
`Selector` should be speficied using a dictionary, where the key is the name
of the parameter whose value the user wishes to override. For example, if I
want to use a simple variator but I want to use the `SBX` variator I would
create a structure similar to the one depicted below. Since I do not specify
other entries in the dictionary, the solver will use the default values for the
SBX variator.

julia> variator = Dict(:name => SBX);

However, if the user desires to customize the `SBX` variator, then he would have
to create other entries in the above dictionay, like so:

julia> variator =  Dict(:name => SBX,
                        :probability => 0.5,
                        :distribution_index => 8);
"""
struct PlatypusSolver <: AbstractSolver
    algorithm::Type
    algorithm_params::Dict{Symbol,Any}
    max_evaluations::Int
    function PlatypusSolver(algorithm::Type; algorithm_params=Dict{Symbol, Any}(), max_eval=100)
        check_arguments(PlatypusSolver, algorithm, algorithm_params, max_eval)
        new(algorithm, algorithm_params, max_eval)
    end
end

get_algorithm(solver::PlatypusSolver) = solver.algorithm
get_max_evaluations(solver::PlatypusSolver) =  solver.max_evaluations
get_algorithm_params(solver::PlatypusSolver) = solver.algorithm_params
get_algorithm_param(solver::PlatypusSolver, param::Symbol, default=nothing) =
  get(get_algorithm_params(solver), param, default)

struct PlatypusSolverException <: Exception
  param::Symbol
  value::Any
  reason::Any
  fix::String
end
Base.showerror(io::IO, e::PlatypusSolverException) =
  print(io, "$(string(e.param)) with value $(string(e.value)) of type $(typeof(e.value)) is $(e.reason).$(e.fix)")

# Argument Verification --------------------------------------------------
"Verifies if the mandatory parameters of a certain `func` are all present in `params`"
function mandatory_params_satisfied(func::Type, params; param_exceptions=[nothing])
  func_params = filter(x -> !(x in param_exceptions), Platypus.mandatory_params(func))

  params = keys(params)
  satisfied_params = map(p -> p in params, func_params)
  all(satisfied_params)
end

"""
  check_arguments(PlatypusSolver, algorithm, algorithm_params, max_eval)

Verifies if the solver is properly defined. The solver will be properly defined
if the algorithm is supported by the platypus solver, if the mandatory
parameters for a certain algorithm are specified, andi f the number of maximum
evaluations in greater than 0.
"""
function check_arguments(solver::Type{PlatypusSolver}, algorithm, algorithm_params, max_eval)
  if  max_eval < 1
    throw(PlatypusSolverException(
            :max_eval,
            e,
            " is invalid. The number of maximum evaluations should be positive",
            "Specify a number of evaluations greater than zero."))
  elseif !mandatory_params_satisfied(algorithm, algorithm_params, param_exceptions=[:problem])
    throw(PlatypusSolverException(
            :algorithm_params,
            algorithm_params,
            " not sufficient. The algorithm's mandatory parameters were not supplied.",
            "Ensure you have specified all $(keys(algorithm_params)) except the parameter problem."))
  end
end

"""
  check_params(solver, model)

Verifies if the conditions on the model and the solver are compliant. Depending
on the algorithm different rules must be satisfied. For instance, in the case
of mixed type variable problems, a variator must be specified.
"""
function check_params(solver::PlatypusSolver, model::Model)
  if ismixedtype(model) && get_algorithm_param(solver, :variator) == nothing
    throw(PlatypusSolverException(
            :variables,
            variables(model),
            "is a mixed integer problem",
            "Please specify one `variator` capable of handling both Integer and Real variables"))
  elseif get_algorithm(solver) in (Platypus.GeneticAlgorithm, Platypus.EvolutionaryStrategy) &&
    nobjectives(model) > 1
      throw(PlatypusSolverException(
              :nobjectives,
              nobjectives(model),
              " a multi objective problem (nobjectives > 1)",
              "Specify an Multi Objective Algorithm or reduce the number of objectives to 1"))
  elseif get_algorithm_param(solver, :population_size, -1) > get_max_evaluations(solver)
      throw(PlatypusSolverException(
              :population_size,
              get_algorithm_param(solver, :population_size),
              "is greater than the population size with value $(get_algorithm_param(solver, :population_size)).",
              "Population size should be smaller or equal to max_evaluations."))
  elseif get_algorithm_param(solver, :offspring_size, -1) > get_max_evaluations(solver)
      throw(PlatypusSolverException(
              :max_evaluations,
              get_algorithm_param(solver, :offspring_size),
              "is greater than the offpsring size with value $(get_algorithm_param(solver, :offspring_size)).",
              "Offpsring size should be smaller or equal to max_evaluations."))
  end
end

"""
  The steps to solve a problem are the following:

  **Step 1**. Define the [`Model`](@ref)
    1. Specify the variables, i.e., identify their type (Real or Integer), the
    maximum and minimum values of each, and, if any, the initial value.

    julia> vars = [IntVariable(0, 20), IntVariable(-20, 0)]; # No initial value

    2. Specify the objectives, i.e., identify the objective function, their sense
    (e.g. MIN or MAX), and optionally their coefficient.

    julia> objs = [Objective(x -> x[1] + x[2], :MIN)]; # No coefficient

    3. If any, specify the constraints, i.e., identify the function, the coefficient
    and the comparison operator (e.g., ≤, <, ==, !=, >, ≥). Note that the comparison
    is against 0 and, therefore, the function should account for that.

    julia> cnstrs = [Constraint(x-> x[1] - 2, <=)]; # Equivalent to x[1] <= 2

    4. Create the model, using the variables, objectives and constraints defined
    previously.

    julia> model = Model(vars, objs, constrs);

  **Step 2**. Specify and configure the Solver to be used
    1. Specify the solver to be used. In the case of the PlatypusSolver, this
    requires specifying the type of the algorithm to be used, any mandatory
    algorithm parameter and the maximum number of evaluations. If the user wants
    to customize the algorithm to be used it can also redefine any algorithm
    parameter.

    julia> a_type = NSGAII;
    julia> a_params = Dict(:population_size => 10);
    julia> solver = PlatypusSolver(a_type, max_eval=90, algorithm_params=a_params);

  **Step 3**. Solve it!
    1. Pass the `model` and the `solver` that you have just defined and wait for
    the results.

    julia> res = solve(solver, model);
"""
function solve(solver::PlatypusSolver, model::Model)
    check_params(solver, model)

    problem = convert(Platypus.Problem, model)

    algorithm_type = get_algorithm(solver)
    evals = get_max_evaluations(solver)
    extra_params = get_algorithm_params(solver)

    # Filter by the fields that are acceptable for the specified algorithm_type
    params = union( Platypus.mandatory_params(algorithm_type),
                    Platypus.optional_params(algorithm_type))
    extra_params = convert_params(filter(p -> first(p) in params, extra_params))
    # Create the algorithm and solve it
    algorithm = algorithm_type(problem; dict2expr(extra_params)...)

    sols = Platypus.solve(algorithm, max_eval=evals)
    convert(Vector{Solution}, sols)
end
