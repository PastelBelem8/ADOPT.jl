module Platypus

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
abstract type PlatypusWrapped end

PyCall.PyObject(x::PlatypusWrapped) = x.pyo

# Create a dictionary to contain the PyObject->Julia_Wrapper type associations
const pre_type_map = []
const type_map = Dict{PyObject,Type}()

"""
  @pytype <name> <class>

Creates the corresponding mutable Julia class and its constructors.
"""
macro pytype(name, class)
  quote
    struct $(esc(name)) <: PlatypusWrapped
      pyo::PyObject
      $(esc(name))(pyo::PyObject) = new(pyo)

      # Create Constructor
      function $(esc(name))(args...; kwargs...)
        new(pycall(platypus[$(QuoteNode(name))], PyObject, args...;kwargs...))
      end
    end
    # Associate PyObject <name> with the Julia Type <name>
    push!(pre_type_map, ($class, $name))
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

"""
  pyattr(class, jl_method, py_method)

 Uses Python's reflection capabilities to add methods to Python classes.
"""
function pyattr(class, jl_method, py_method)
  quote
    function $(esc(jl_method))(pyt::$class, args...; kwargs...)
      #TODO - Fix the arguments (conversion to Python)
      n_args = args
      method = pyt.pyo[$(string(py_method))]
      pyo = pycall(method, PyObject, n_args...; kwargs...)
      wrapped = platypus_wrap(pyo)
    end
  end
end

pyattr(class, method) = pyattr(class, method, method)

macro pyattr(class,  method)
  pyattr(class, method)
end

macro pyattr(class, method, orig_method)
  pyattr(class, method, orig_method)
end

"""
  pyattr_set(types, methods...)

For each Julia Type `T<:PlatypusWrapped` in `types` and each method `m`
in `methods`, define a new function `m(t::T, args...)` that delegates
to the underlying pyobject wrapped by `t`.
"""
function pyattr_set(classes, methods...)
  for class in classes
    for method in methods
      @eval @pyattr($class, $method)
    end
  end
end

# Redefine custom printing methods
function Base.show(io::IO, pyv::PlatypusWrapped)
  s = pyv.pyo[:__str__]()
  println(io, s)
end

@pytype NSGAII ()->platypus[:NSGAII]
@pytype Problem ()->platypus[:Problem]
@pytype NSGAII ()->platypus[:NSGAII]
@pytype Solution ()->platypus[:Solution]

import Base.run
pyattr_set([NSGAII], :run)

# Tests
# p = Problem(1, 2, 0, (x) -> [x[1]^2, (x[2]-2)^2])
# a = NSGAII(p)
# run(a, 1000)

end # Module
