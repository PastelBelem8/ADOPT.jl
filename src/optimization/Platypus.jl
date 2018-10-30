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
macro pytype(name)
  pyclass = () -> platypus["$name"] # For Unwrapping

  constructor_name = platypus_classname(name)
  constructor_symbol = esc(Symbol(constructor_name))

  quote
    struct $(constructor_symbol) <: PlatypusWrapper
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
@pytype Real
@pytype Integer

# Problem / Model
@pytype Constraint
@pytype Solution
@pytype Problem
@pytype_setters Problem constraints directions "function" types

# Algorithms
# Single Objective Algorithms
@pytype GeneticAlgorithm
@pytype EvolutionaryStrategy

# Multi Objective Algorithms
@pytype CMAES
@pytype EpsMOEA
@pytype EpsNSGAII
@pytype GDE3
@pytype IBEA
@pytype MOEAD
@pytype NSGAII
@pytype NSGAIII
@pytype PAES
@pytype PESA2
@pytype OMOPSO
@pytype SMPSO
@pytype SPEA2

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

# Algorithm Routines --------------------------------------------------------
platypus_algorithms =
    Dict(   # Single Objective
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
            :SPEA2      => _SPEA2 )

"Returns whether a specific `algorithm` is currently supported"
supports(algorithm::Symbol) = haskey(platypus_algorithms[algorithm])

"Returns an ordered list of the optimization algorithms currently supported"
supported_algorithms() = sort(collect(keys(platypus_algorithms)))

"Creates an algorithm instance"
Algorithm(algorithm::Symbol, problem::_Problem; kwargs...) =
    platypus_algorithms[algorithm](problem; kwargs...)


function solve(algorithm::PlatypusWrapper, max_eval::Int; kwargs...)
    # run(algorithm; kwargs...)
    algorithm.pyo[:run](max_eval)
    res = algorithm.pyo[:result]
    platypus_wrap(res)
end



# TODO
# Reflectir sobre se é justificavel permitir que a partir de um construtor de
# Problema se possa criar a instancia em Python ou não.
# Definir macro que cria automaticamente p/ cada algoritmo o método run associado.


# using PyCall
# (Platypus.platypus[:NSGAII])
# print(PyDict(PyCall.inspect[:signature](Platypus.platypus[:NSGAII])[:parameters])["population_size"][:default])
#
# function inspect_signature(func::Symbol)
#     params = PyDict(PyCall.inspect[:signature](Platypus.platypus[func])[:parameters])
#     res = []
#     for (pname, param) in params
#       res = vcat((pname, params[pname][:default]), res)
#     end
#     res
# end
# inspect_signature(:NSGAII)
end # Module
