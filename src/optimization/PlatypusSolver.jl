# Using
using .Platypus

# Imports
import Base: convert

"Converts a dictionary to an array of expressions to be passed as kwargs"
dict2expr(dict::Dict) = :($([Expr(:(=), kv[1], kv[2]) for kv in dict]))

# Converter routines ----------------------------------------------------
# These routines provide the interface between the solver and the
# Platypus library.
# -----------------------------------------------------------------------
platypus_function(objectives, constraints)::Function =
    length(constraints) > 0  ? ((x...) -> return [func(o)(x...) for o in objectives],
                                                 [func(c)(x...) for c in constraints]) :
                               ((x...) -> return [func(o)(x...) for o in objectives])

# ------------------------------------------------------------------------
# Problem Converter Routines
# ------------------------------------------------------------------------
function convert(::Type{Platypus.Problem}, m::Model)
  # 1. Create Base Problem
  nvars, nobjs, nconstrs = nvariables(m), nobjectives(m), nconstraints(m)
  problem = Platypus.Problem(nvars, nobjs, nconstrs)

  # 2. Refine the Problem instance
  # 2.1. Convert types
  Platypus.set_types!(problem, [convert(Platypus.PlatypusWrapper, v)
                                        for v in variables(m)])

  # 2.2. Convert Constraints
  constrs = []
  if nconstrs > 0
    constrs = constraints(m)
    Platypus.set_constraints!(problem, [convert(Constraint, c)
                                                for c in constrs])
  end

  # 2.3. Convert Objective Function
  objs = objectives(m)
  Platypus.set_directions!(problem, directions(objs))
  Platypus.set_function!(problem, platypus_function(objs, constrs))
  problem
end
convert(::Platypus.Constraint, c::Constraint) =
    Platypus.Constraint(string(operator(c)), 0)

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
  variables = Platypus.get_variables(s)
  objectives = Platypus.get_objectives(s)
  constraints = Platypus.get_constraints(s)
  constraint_violation = Platypus.get_constraint_violation(s)
  feasible = Platypus.get_feasible(s)
  evaluated = Platypus.get_evaluated(s)

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
# Platypus Solver --------------------------------------------------------
"""
    PlatypusSolver
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
  reason::Type
  fix::Any
end
Base.showerror(io::IO, e::PlatypusSolverException) =
  print(io, "$(e.param) with value $(e.value) of type $(typeof(e.value)) is $(e.reason).$(e.fix)")

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
            "is a mixed integer problem.",
            "Please specify one `variator` capable of handling both Integer and Real variables."))
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
    # algorithm = algorithm_type(problem, :($(dict2expr(extra_params)...)))
    algorithm = algorithm_type(problem)
    sols = Platypus.solve(algorithm, max_eval=evals)
end


#  Test
# m = Model([IntVariable(0, 100, 2), IntVariable(0, 100, 2)],[Objective(x -> x[1] + x[2])])
# p = convert(Platypus.Problem, m)
# a = Platypus.Algorithm(SPEA2, p)
# convert(Vector{Solution}, Platypus.solve(a, max_eval=3))
# var = Platypus.solve(a, 100)

# sols = Platypus.solve(a, 100, unique_objectives=false)
# Platypus.all_results(a)
# Platypus.get_unique(sols)
# Platypus.get_feasible(sols)
# Platypus.get_nondominated(sols)
# x = convert(Solution, sol)
#
# Platypus.get_feasible(sols[1])
#
# variator = Dict(:name => SBX)

# algorithm_params = Dict(:population_size => 30,  :variator => variator)
# solver = PlatypusSolver(SPEA2, max_eval=90, algorithm_params=algorithm_params)
# model = Model([IntVariable(10, 13, 12), IntVariable(-10, 10, 2)],[Objective(x -> (x[1] * x[2]) ^ 2), Objective(x -> x[1] - x[2])])
# sols = solve(solver, model)
# convert(Vector{Main.Solution}, sols)
