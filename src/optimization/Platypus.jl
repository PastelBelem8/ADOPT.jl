module Platypus

import Base: show

using PyCall

const platypus = PyNULL()

function __init__()
   copy!(platypus, pyimport_conda("platypus", "platypus-opt", "conda-forge"))
   for (platypus_expr, julia_type) in pre_type_map
       type_map[platypus_expr()] = julia_type
    end
end
# Platypus Wrappers ---------------------------------------------------------
# This implementation is inspired in Pandas.jl implementation. Using Python's
# metaprogramming capabilities we are able to mimic the existing data
# structures in Julia and provide a similar interface.
# ---------------------------------------------------------------------------

# Wrapper type that gathers the behavior of Platypus entities
abstract type PlatypusWrapper end
abstract type PlatypusAlgorithm <: PlatypusWrapper end

PyCall.PyObject(x::PlatypusWrapper) = x.pyo

# Create a dictionary to contain the PyObject->Julia_Wrapper type associations
const pre_type_map = []
const type_map = Dict{PyObject,Type}()

platypus_class(name::String) = Symbol("_$(name)")
platypus_class(name::Symbol) = Symbol("_$(string(name))")
platypus_classname(name::String) = "_$(name)"
platypus_classname(name::Symbol) = "_$(name)"

"""
  @pytype name pyclass

Creates the corresponding Julia class and its constructors.
"""
macro pytype(name, parent=:PlatypusWrapper)
  parent = Symbol(parent)
  pyclass = () -> platypus["$name"] # For Unwrapping

  constructor_name = platypus_classname(name)
  constructor_symbol = esc(Symbol(constructor_name))

  quote
    struct $(constructor_symbol) <: $(esc(parent))
      pyo::PyObject
      $(constructor_symbol)(pyo::PyObject) = new(pyo)

      # Create Constructor
      function $(constructor_symbol)(args...; kwargs...)
        platypus_method = platypus[$(QuoteNode(name))]
        new(pycall(platypus_method, PyObject, args...; kwargs...))
      end
    end

    # Associate PyObject <name> with the Julia Type <name>
    push!(pre_type_map, ($pyclass, $constructor_symbol))
  end
end

# Representation
function Base.show(io::IO, pyv::PlatypusWrapper)
  s = pyv.pyo[:__str__]()
  println(io, s)
end

"""
  set_attr(class, name, type)

Creates a setter for the specified attribute `name` with default type `type` if
  specified.
"""
function set_attr(class, name, typ=nothing)
  param = typ == nothing ? (:t) : (:(t::$(typ)))
  setter_name = Symbol("set_$(name)!")

  quote
    $(esc(setter_name))(o :: $(esc(class)), $(param)) =
        o.pyo[$(QuoteNode(name))] = t
  end
end

macro pytype_setters(class, attrs...)
  class = platypus_class(class)

  get_name = (attr) -> isa(attr, Expr) ? attr.args[1] : attr
  get_type = (attr) -> isa(attr, Expr) ? attr.args[2] : nothing

  param_names = map(get_name, attrs)
  param_types = map(get_type, attrs)

  setters = map((name, typ) -> set_attr(Symbol(class), name, typ), param_names, param_types)

  quote
    $(setters...)
  end
end

macro pytypes_setters(classes::Expr, attrs...)
  for class in classes.args
    @eval(@pytype_setters($class, $(attrs...)))
  end
end

"""
  get_attr(class, name, type)

Creates a getter for the specified attribute `name`.
"""
function get_attr(class, name)
  getter_name = Symbol("get_$(name)")

  quote
    $(esc(getter_name))(o :: $(esc(class))) =
        platypus_wrap(o.pyo[$(QuoteNode(name))])
  end
end

macro pytype_getters(class, attrs...)
  class = platypus_class(class)
  getters = map(name -> get_attr(Symbol(class), name), attrs)

  quote
    $(getters...)
  end
end

"""
  platypus_wrap(pyo::PyObject)

Wraps an instance of platypus' Python class in the Julia type which corresponds
to that class.
"""
function platypus_wrap(pyo::PyObject)
  for (pyt, pyv) in type_map
    if pyisinstance(pyo, pyt)
      return pyv(pyo)
    end
  end

  convert(PyAny, pyo)
end

platypus_wrap(x::Union{AbstractArray, Tuple}) = [platypus_wrap(_) for _ in x]
platypus_wrap(pyo) = pyo

# Platypus Julia's Proxies --------------------------------------------------
# Types
# @pytype_iterable FixedLengthArray
@pytype Real
@pytype Integer

# Problem / Model
@pytype Constraint

@pytype Solution
@pytype_getters Solution variables objectives constraints constraint_violation feasible evaluated
function get_variables(solution::_Solution; toDecode::Bool=true)
  if toDecode
    types = solution.pyo[:problem][:types]
    vars = solution.pyo[:variables]
    decoded_vars = Vector()
    for t in types, v in size(vars, 1)
      decoded_vars = vcat(t[:decode](vars[v,:]), decoded_vars)
    end
    decoded_vars
  else
    get_variables(solution)
  end
end

@pytype Problem
@pytype_getters Problem nvars nobjs nconstrs "function" types directions constraints
@pytype_setters Problem constraints directions "function" types

# Algorithms
# Single Objective Algorithms
@pytype GeneticAlgorithm PlatypusAlgorithm
@pytype EvolutionaryStrategy PlatypusAlgorithm

# Multi Objective Algorithms
@pytype CMAES PlatypusAlgorithm
@pytype EpsMOEA PlatypusAlgorithm
@pytype EpsNSGAII PlatypusAlgorithm
@pytype GDE3 PlatypusAlgorithm
@pytype IBEA PlatypusAlgorithm
@pytype MOEAD PlatypusAlgorithm
@pytype NSGAII PlatypusAlgorithm
@pytype NSGAIII PlatypusAlgorithm
@pytype PAES PlatypusAlgorithm
@pytype PESA2 PlatypusAlgorithm
@pytype OMOPSO PlatypusAlgorithm
@pytype SMPSO PlatypusAlgorithm
@pytype SPEA2 PlatypusAlgorithm

@pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,NSGAII,EpsMOEA,GDE3, SPEA2, IBEA, PESA2], population_size::Int)
# @pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,NSGAII,CMAES], offspring_size::Int)
# @pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,NSGAII,EpsMOEA,GDE3,SPEA2,MOEAD,NSGAIII,IBEA,PAES,PESA2,PSO,OMOPSO,SMPSO,CMAES], generator)
# @pytypes_setters([GeneticAlgorithm,NSGAII,EpsMOEA, NSGAIII], selector)
# @pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,NSGAII,EpsMOEA,GDE3,SPEA2,MOEAD,NSGAIII,IBEA,PAES,PESA2], generator)
# @pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy, IBEA], comparator)
# @pytypes_setters([GeneticAlgorithm,SPEA2, PSO, OMOPSO,SMPSO], dominance)
# @pytypes_setters([NSGAII], archive)
# @pytypes_setters([EpsMOEA,OMOPSO,CMAES], epsilons)
# @pytypes_setters([SPEA2,MOEAD], k)

# Generators
@pytype RandomGenerator

# Selectors
@pytype TournamentSelector

# Variators
@pytype SBX
@pytype HUX
@pytype PCX
@pytype PMX
@pytype SPX
@pytype SSX
@pytype UNDX
# Variators - Mutation
@pytype Insertion
@pytype NonUniformMutation
@pytype PM
@pytype Replace
@pytype Swap
@pytype UM
@pytype UniformMutation

# Algorithm Routines --------------------------------------------------------
platypus_algorithms =
    Dict(
      # Single Objective
      :GeneticAlgorithm     => _GeneticAlgorithm,
      :EvolutionaryStrategy => _EvolutionaryStrategy,
      # Multi Objective
      :CMAES      => _CMAES,
      :EpsMOEA    => _EpsMOEA,
      :EpsNSGAII  => _EpsNSGAII,
      :GDE3       => _GDE3,
      :IBEA       => _IBEA,
      :MOEAD      => _MOEAD,
      :NSGAII     => _NSGAII,
      :NSGAIII    => _NSGAIII,
      :PAES       => _PAES,
      :PESA2      => _PESA2,
      :OMOPSO     => _OMOPSO,
      :SMPSO      => _SMPSO,
      :SPEA2      => _SPEA2
   )

"Returns whether a specific `algorithm` is currently supported"
supports(algorithm::Symbol) = haskey(platypus_algorithms[algorithm])

"Returns an ordered list of the optimization algorithms currently supported"
supported_algorithms() = sort(collect(keys(platypus_algorithms)))

"Creates an algorithm instance and associates it to the specified `problem`"
Algorithm(algorithm::Union{Symbol, String}, problem::_Problem; kwargs...) =
    platypus_algorithms[algorithm](problem; kwargs...)

function solve(algorithm_name::Symbol, problem::_Problem; max_eval::Int, algorithm_params...)
  algorithm = Algorithm(algorithm_name, problem, algorithm_params...)
  algorithm.pyo[:run](max_eval)
  platypus_wrap(results(algorithm))
end

all_results(algorithm::PlatypusAlgorithm) =
  platypus_wrap(algorithm.pyo[:result])
get_unique(solutions::Vector{_Solution}) =
  platypus_wrap(map(platypus[:unique], solutions))
get_feasible(solutions::Vector{_Solution}) =
  platypus_wrap(filter(s -> get_feasible(s), solutions))
get_nondominated(solutions::Vector{_Solution}) =
  platypus_wrap(platypus[:nondominatd](solutions))

function results(algorithm::PlatypusAlgorithm; unique::Bool=true,
                  feasible::Bool=true, nondominated::Bool=true)
  results = all_results(algorithm)
  results = unique ? get_unique(results) : results
  results = feasible ? get_feasible(results) : results
  results = unique ? get_nondominated(results) : results
  results
end
# Reflection Capabilities -----------------------------------------------
# These methods will be used to generate the running methods for each algorithm
# according to the implementation. Two methods will be generated by each class.
# One providing only the mandatory arguments and other providing the ability
# to change other parameters.
# -----------------------------------------------------------------------
"Uses the reflection capabilities of python to parse a function's signature and retrieves its arguments"
function inspect_signature(pyfunc::Symbol)
    params = PyDict(PyCall.inspect[:signature](Platypus.platypus[pyfunc])[:parameters])
    res = []
    for (pname, param) in params
      res = vcat((pname, params[pname][:default]), res)
    end
    res
end

"Returns a tuple subdiving the mandatory arguments from the optional arguments"
function get_parameters(pyfunc::Symbol)
  iskwargs = name -> name == "kwargs"
  isempty = value -> value == PyCall.inspect[:Parameter][:empty]
  isMandatory = param -> !iskwargs(param[1]) && isempty(param[2])

  args = inspect_signature(pyfunc)
  mandatory_args = map(arg -> arg[1], filter(isMandatory, args))
  optional_args = map(arg -> (arg[1], platypus_wrap(arg[2])), filter(!isMandatory, args))

  (mandatory_args, optional_args)
end

mandatory_params(name::Symbol) = map(Symbol, get_parameters(name)[1])
optional_params(name::Symbol) = map(a -> Symbol(a[1]), get_parameters(name)[2])

end # Module
