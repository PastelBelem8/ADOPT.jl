module Platypus

import Base: convert, show

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

"""
  @pytype name pyclass

Creates the corresponding Julia class and its constructors.
"""
macro pytype(name, pyclass, fields...)
  constructor_name = esc(Symbol("_$name"))

  # setter_names = map(name -> "set_$(name)!(pyo::$constructor_name, $name) =
  # pyo[:$(name)] = $name", fields)

  quote
    struct $(constructor_name) <: PlatypusWrapper
      pyo::PyObject
      $(constructor_name)(pyo::PyObject) = new(pyo)

      # Create Constructor
      function $(constructor_name)(args...; kwargs...)
        platypus_method = ($pyclass)()
        new(pycall(platypus_method, PyObject, args...; kwargs...))
      end

      # $(setter_names...)
    end

    # Associate PyObject <name> with the Julia Type <name>
    push!(pre_type_map, ($pyclass, $constructor_name))
  end
end

# Representation
function Base.show(io::IO, pyv::PlatypusWrapper)
  s = pyv.pyo[:__str__]()
  println(io, s)
end

# Platypus Julia's Proxies ---------------------------------------------------
# Types
@pytype Type ()->platypus[:Type]
@pytype Real ()->platypus[:Real]
@pytype Integer ()->platypus[:Integer]

# Problem / Model
@pytype Constraint ()->platypus[:Constraint]
@pytype Solution ()->platypus[:Solution]
@macroexpand @pytype Problem ()->platypus[:Problem] constraints directions "function" types

function convert(t::_Problem, m::Model)
  # 1. Create Base Problem
  nvars, nobjs, nconstrs = nvariables(m), nobjectives(m), nconstraints(m)
  problem = _Problem(nvars, nobjs, nconstrs)

  # 2. Refine the Problem instance
  # 2.1. Convert types --------------------------------------------------
  set_types!(problem, [convert(_Type, v) for v in variables(m)])

  # 2.2. Convert Constraints ---------------------------------------------
  constrs = []
  if nconstrs > 0
    constrs = constraints(m)
    set_constraints!(problem, [convert(_Constraint, c) for c in constraints(m)])
  end

  # 2.3. Convert Objective Function --------------------------------------
  objectives = objectives(m)
  set_directions!(problem, directions(objectives))
  set_function!(problem, (x...) -> [func(o)(x...) for o in objectives],
                                   [func(c)(x...) for c in constrs])
end

convert(t::_Constraint, c::Constraint) = _Constraint(string(operator(c)), 0)
convert(t::_Type, variable::IntVariable) =
    _Integer(lower_bound(variable), upper_bound(variable))
convert(t::_Real, variable::RealVariable) =
    _Real(lower_bound(variable), upper_bound(variable))


# TODO
# Reflectir sobre se é justificavel permitir que a partir de um construtor de Problema se possa criar a instancia em Python ou não.
# Passar do simbolo do algoritmo p/ objecto em Python
# Testar se corre um problema
# Wrap de _Solution (Python) em Solution (Julia)
# Tests
# p = _Problem(1, 2, 0, (x) -> [x[1]^2, (x[2]-2)^2])
# a = NSGAII(p)
# run(a, 1000)


#
# """
#   platypus_wrap(pyo::PyObject)
#
# Wraps an instance of platypus' Python class in the Julia type which corresponds
# to that class.
# """
# function platypus_wrap(pyo::PyObject)
#   for (pyt, pyv) in type_map
#     if pyisinstance(pyo, pyt)
#       return pyv(pyo)
#     end
#   end
#   convert(PyAny, pyo)
# end
#
# platypus_wrap(x::Union{AbstractArray, Tuple}) = [platypus_wrap(_) for _ in x]
# platypus_wrap(pyo) = pyo

end # Module
