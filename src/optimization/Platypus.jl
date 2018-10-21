module Platypus

using PyCall

const platypus_raw = PyNULL()

const types_map = []

# Only called once, and only after all statements in the module have been executed
function __init__()
  copy!(platypus_raw, pyimport_conda("platypus", "platypus-opt", "conda-forge"))

   types_map = Dict(Union{Real} => platypus_raw[:Real],
                    Int  => platypus_raw[:Integer])
end

# Platypus Wrappers ---------------------------------------------------------
# This implementation is inspired in Pandas.jl implementation. Using Python's
# metaprogramming capabilities we are able to mimic the existing data
# structures in Julia and provide a similar interface.
# However, we do not want to expose all the existing types in platypus.
# ---------------------------------------------------------------------------

# Wrapper type gathering the behavior of Platypus' objects
abstract type PlatypusWrapped end

PyCall.PyObject(x::PlatypusWrapped) = x.pyo

"""
  @pytype name supertype pypackage [field1, field2, ...]

Defines a Julia composite type, subtype of `supertype`, that wrap the class
`name` from the python package `pypackage`. When specified, the fields are
converted to the appropriate subtype in python.

The conversion mentioned in the previous paragraph relies on the mapping
existing in an external data structure.
"""
macro deftype(name, supertype, fields...)
  name_str = string(name) |> lowercase
  rename_str = "platypus_$(name_str)"

  python_constructor = platypus_raw[name]
  constructor_name = esc(Symbol("_$(rename_str)"))
  public_constructor_name = esc(Symbol("create_$(rename_str)"))

  mk_conversion_call(field) = Expr(:call, Symbol("create_$(string(field))!"),
    :tpyo, :(getindex(kwargs, :types))) #string($(esc(Symbol(field)))))))
  conversion_calls = map(mk_conversion_call, fields)

  quote
    export $(public_constructor_name)
    struct $(constructor_name) <: $(esc(supertype))
      pyo::PyObject
      $(constructor_name)(pyo::PyObject) = new(pyo)

      function $(constructor_name)(args...; kwargs...)
        tpyo = pycall($(python_constructor), PyObject, args...)
        $(conversion_calls...)
        new(tpyo)
      end
    end

    $(public_constructor_name)(args...; kwargs...) = $(constructor_name)(args...; kwargs...)
  end
end

# Redefine custom printing methods
function Base.show(io::IO, pyv::PlatypusWrapped)
  s = pyv.pyo[:__str__]()
  println(io, s)
end

# Platypus Conversion Routines ----------------------------------------------

function create_types!(o::PyObject, types::Array{T, 1}) where T
  λ(t) = pycall(types_map[typeof(t[1])], PyObject, (t...))
  converted_types = PyObject(map(λ, types))
  o[:types] = PyVector(converted_types)
end

function create_constraints!(o::PyObject, constraints::Array{T, 1}) where T
  throw(ErrorException("not implemented yet."))
end



# # #
# @macroexpand @deftype Problem PlatypusWrapped types constraints
@deftype Problem PlatypusWrapped types #constraints
@deftype Solution PlatypusWrapped
@deftype NSGAII PlatypusWrapped types #constraints
@deftype Problem PlatypusWrapped types #constraints
@deftype Problem PlatypusWrapped types #constraints

# p = create_platypus_problem(3, 2, 0; types=[(0, 2), (3, 4)])


end # Module
