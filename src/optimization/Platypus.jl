module Platypus

using PyCall

const platypus = PyNULL()

function __init__()
  copy!(platypus, pyimport_conda("platypus", "platypus-opt", "conda-forge"))
end

# Platypus Wrappers --------------------------------------------------------- 
# This implementation is inspired in Pandas.jl implementation. Using Python's
# metaprogramming capabilities we are able to mimic the existing data
# structures in Julia and provide a similar interface.
# ---------------------------------------------------------------------------

# Wrapper type gathering the behavior of Platypus' objects
abstract type PlatypusWrapped end

PyCall.PyObject(x::PlatypusWrapped) = x.pyo


macro pytype(name, wClass, pkg, wFields...)
  pkgStr = string(pkg)  
  nameStr = string(name)
  renamedStr = "$(pkgStr)$(nameStr)" 
 
  constructorName = esc(Symbol("_$(renamedStr)"))
  constructorName_wrapper = esc(Symbol("create_$(renamedStr)")) 
  pythonConstructor = platypus[name]  
  mk_unwrap_call(field) = Expr(:call, Symbol("create_$(string(field))"), :tpyo, :(kwargs[QuoteNode($(esc(field)))]))
  unwrap_calls = map(mk_unwrap_call, wFields)
  
  quote
    export $(constructorName_wrapper) 
    struct $(constructorName) <: $(esc(wClass))
      pyo::PyObject 
      $(constructorName)(pyo::PyObject) = new(pyo)
      
      function $(constructorName)(args...; kwargs...)
        tpyo = pycall(pythonConstructor, PyObject, args...;kwargs...)
        $(unwrap_calls...)                                 
        new(tpyo)
      end
    end
    
    function $(constructorName_wrapper)(args...; kwargs...)
       $(constructorName)(args...; kwargs...)
    end
  end
end

end # Module
