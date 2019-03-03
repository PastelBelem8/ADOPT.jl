module Platypus

import Base: show

using PyCall

const platypus = PyNULL()

function __init__()
   copy!(platypus, pyimport_conda("platypus", "platypus-opt", "conda-forge"))
   version = VersionNumber(platypus[:__version__])
   @info("Your Python's Platypus has version $version.")
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

"""
  @pytype name pyclass

Creates the corresponding Julia class and its constructors.
"""
macro pytype(name, parent=:PlatypusWrapper)
  parent = Symbol(parent)
  constructor_symbol = esc(Symbol(name))

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
function get_attr(class, name, out)
  getter_name = Symbol("get_$(name)")

  quote
    $(esc(getter_name))(receiver :: $(esc(class))) =
        $(esc(out))(receiver.pyo[$(QuoteNode(name))])
  end
end

macro pytype_getters(class, attrs...)
  attr_names = map(attr -> attr.args[1], attrs)
  attr_outs = map(attr -> attr.args[2], attrs)

  getters = map((name, out) -> get_attr(Symbol(class), name, out), attr_names, attr_outs)

  quote
    $(getters...)
  end
end

# Platypus Julia's Proxies --------------------------------------------------
# Types
@pytype Real
@pytype Integer

# Problem / Model
@pytype Constraint

@pytype Solution
@pytype_getters(Solution,
                objectives::PyVector,
                constraints::PyVector,
                constraint_violation::Base.Real,
                feasible::Bool,
                evaluated::Bool)
function get_variables(solution::Solution; toDecode::Bool=true)
  vars = solution.pyo[:variables]
  if toDecode
    types = solution.pyo[:problem][:types]
    decoded_vars = Vector()
    for i in 1:size(vars, 1)
      decoded_vars = vcat(decoded_vars, types[i][:decode](vars[i,:]))
    end
    decoded_vars
  else
    vars
  end
end

@pytype Problem
@pytype_getters(Problem,
                nvars::Int,
                nobjs::Int,
                nconstrs::Int,
                "function"::pyfunction,
                types::PyVector,
                directions::PyVector,
                constraints::PyVector)
@pytype_setters Problem directions "function"
set_constraints!(o::Platypus.Problem, constraints) =
  o.pyo[:constraints][:__setitem__]((PyCall.pybuiltin(:slice)(nothing, nothing, nothing)), constraints)

set_types!(o::Platypus.Problem, types) =
  o.pyo[:types][:__setitem__]((PyCall.pybuiltin(:slice)(nothing, nothing, nothing)), types)

# Algorithms
# Single Objective Algorithms
@pytype GeneticAlgorithm PlatypusAlgorithm
@pytype EvolutionaryStrategy PlatypusAlgorithm
export GeneticAlgorithm, EvolutionaryStrategy

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
export CMAES, EpsMOEA, EpsNSGAII, GDE3, IBEA, MOEAD, NSGAII, NSGAIII,
        PAES, PESA2, OMOPSO, SMPSO, SPEA2

@pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,NSGAII,EpsMOEA,GDE3, SPEA2, IBEA, PESA2], population_size::Int)
@pytypes_setters([GeneticAlgorithm,EvolutionaryStrategy,CMAES], offspring_size::Int)
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
export RandomGenerator

# Selectors
@pytype TournamentSelector
export TournamentSelector

# Variators
abstract type PlatypusVariator <: PlatypusWrapper end
abstract type SimpleVariator <: PlatypusVariator end
abstract type CompoundVariator <: PlatypusVariator end

@pytype Variator PlatypusVariator
@pytype CompoundOperator CompoundVariator
@pytype GAOperator CompoundVariator
@pytype SBX SimpleVariator
@pytype HUX SimpleVariator
@pytype PCX SimpleVariator
@pytype PMX SimpleVariator
@pytype SPX SimpleVariator
@pytype SSX SimpleVariator
@pytype UNDX SimpleVariator
export CompoundOperator, GAOperator, SBX, HUX, PCX, PMX, SPX, SSX, UNDX

# Mutations are specific types of Variators
@pytype CompoundMutation CompoundVariator
@pytype Insertion SimpleVariator
@pytype NonUniformMutation SimpleVariator
@pytype PM SimpleVariator
@pytype Replace SimpleVariator
@pytype Swap SimpleVariator
@pytype UM SimpleVariator
@pytype UniformMutation SimpleVariator
export CompoundMutation, Insertion, NonUniformMutation, PM, Replace, Swap, UM,
       UniformMutation


# Algorithm Routines --------------------------------------------------------
get_all_results(algorithm::PlatypusAlgorithm) =
  [Solution(sol) for sol in algorithm.pyo[:result]]
get_unique(solutions::Vector{Solution}; objectives=true) =
  [Solution(sol) for sol in platypus[:unique](solutions, objectives)]
get_feasible(solutions::Vector{Solution}) =
  filter(s -> get_feasible(s), solutions)
get_nondominated(solutions::Vector{Solution}) =
  [Solution(sol) for sol in platypus[:nondominated](solutions)]

function results(algorithm::PlatypusAlgorithm; unique::Bool,
                  unique_objectives::Bool, feasible::Bool, nondominated::Bool)
  results = get_all_results(algorithm)
  results = unique ? get_unique(results, objectives=unique_objectives) : results
  results = feasible ? get_feasible(results) : results
  results = nondominated ? get_nondominated(results) : results
  results
end

"Creates an algorithm instance and associates it to the specified `problem`"
Algorithm(algorithm::Type{T}, problem::Problem; kwargs...) where {T<:PlatypusAlgorithm}=
    algorithm(problem, kwargs...)

function solve(algorithm::T; max_eval::Int, unique::Bool=true,
                unique_objectives::Bool=true, feasible::Bool=true, nondominated::Bool=true) where{T<:PlatypusAlgorithm}
  algorithm.pyo[:run](max_eval)
  results(algorithm, unique=unique, unique_objectives=unique_objectives, feasible=feasible, nondominated=nondominated)
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
  optional_args = map(arg -> (arg[1], arg[2]), filter(!isMandatory, args))

  (mandatory_args, optional_args)
end

mandatory_params(name::Type) = map(Symbol, get_parameters(Symbol(name))[1])
optional_params(name::Type) = map(a -> Symbol(a[1]), get_parameters(Symbol(name))[2])

end # Module
