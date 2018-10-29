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

function make_setter(name, class)
  name_str = string(name)
  setter_name = Symbol("set_$(name_str)!")

  quote
    $(esc(setter_name))(o :: $(class), t) = o.pyo[$(QuoteNode(name_str))] = t
  end
end

"""
  @pytype name pyclass

Creates the corresponding Julia class and its constructors.
"""
macro pytype(name, pyclass, fields...)
  constructor_name = "_$name"
  constructor_symbol = esc(Symbol(constructor_name))

  mk_setters = (field_name) -> make_setter(field_name, constructor_symbol)
  setters = map(mk_setters, fields)

  quote
    struct $(constructor_symbol) <: PlatypusWrapper
      pyo::PyObject
      $(constructor_symbol)(pyo::PyObject) = new(pyo)

      # Create Constructor
      function $(constructor_symbol)(args...; kwargs...)
        platypus_method = ($pyclass)()
        new(pycall(platypus_method, PyObject, args...; kwargs...))
      end
    end

    $(setters...)
    # Associate PyObject <name> with the Julia Type <name>
    push!(pre_type_map, ($pyclass, $constructor_symbol))
  end
end
# @macroexpand @pytype Problem ()->platypus[:Problem] types constraints "function"

# Representation
function Base.show(io::IO, pyv::PlatypusWrapper)
  s = pyv.pyo[:__str__]()
  println(io, s)
end

# Platypus Julia's Proxies ---------------------------------------------------
# Types
@pytype Real ()->platypus[:Real]
@pytype Integer ()->platypus[:Integer]

# Problem / Model
@pytype Constraint ()->platypus[:Constraint]
@pytype Solution ()->platypus[:Solution]
@pytype Problem ()->platypus[:Problem] constraints directions "function" types


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
