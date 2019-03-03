#=
    This file is the identical copy to the one found at
    https://github.com/aptmcl/Khepri.jl/blob/master/src/Parameters.jl
    The repository in question is under constant changes and, therefore,
    we opted for having a local copy
=#
export with, Parameter, LazyParameter

mutable struct Parameter{T}
  value::T
end

(p::Parameter)() = p.value
(p::Parameter)(newvalue) = p.value = newvalue

function with(f, p, newvalue)
  oldvalue = p()
  p(newvalue)
  try
    f()
  finally
    p(oldvalue)
  end
end

with(f, p, newvalue, others...) =
  with(p, newvalue) do
    with(f, others...)
  end

mutable struct LazyParameter{T}
  initializer::Function #This should be a more specific type: None->T
  value::Union{T, Nothing}
end

LazyParameter(T::DataType, initializer::Function) = LazyParameter{T}(initializer, nothing)

(p::LazyParameter)() = p.value === nothing ? (p.value = p.initializer()) : p.value
(p::LazyParameter)(newvalue) = p.value = newvalue

import Base.reset
reset(p::LazyParameter{T}) where {T} = p.value = nothing
