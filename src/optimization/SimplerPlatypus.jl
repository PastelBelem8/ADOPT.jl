module SimplerPlatypus

using PyCall

const platypus_raw = PyNULL()
const type_map = []
const pre_type_map = []

# Only called once, and only after all statements in the module have been executed
function __init__()
  copy!(platypus_raw, pyimport_conda("platypus", "platypus-opt", "conda-forge"))

  for (platypus_expr, julia_type) in pre_type_map
    type_map[platypus_expr()] = julia_type
  end
end

# Wrapper type that gathers the behavior of Platypus entities
abstract type PlatypusWrapped end

PyCall.PyObject(x::PlatypusWrapped) = x.pyo

# TODO - CHANGE THIS TO BE A FUNCTION
macro pyAlgorithm(name, supertype, pyclass)
  quote
    struct $(esc(name)) <: $(esc(supertype))
      pyo::PyObject
      $(esc(name))(pyo::PyObject) = new(pyo)

      # Create Constructor
      function $(esc(name))(args...; kwargs...)
        # FIXME - Change the arguments
        python_method = platypus_raw[$(QuoteNode(name))]
        new(pycall(python_method, PyObject, args...;kwargs...))
      end
    end
    # Associate PyObject <name> with the Julia Type <name>
    push!(pre_type_map, ($pyclass, $name))
  end
end

# CREATE MACRO THAT RECEIVES A BOOLEAN INDICATING WHETHER ONE SHOULD CHANGE THE
# CONSTRUCTOR NAME OR NOT.

platypus_types = [:Problem, :Solution]


platypus_algorithms = [:CMAES, :EpsMOEA, :EvolutionaryStrategy, :GDE3,
                       :GeneticAlgorithm, :IBEA, :MOEAD, :NSGAII, :NSGAIII,
                       :OMOPSO, :PAES, :PESA2, :SMPSO, :SPEA2]

for typ in platypus_algorithms
  @pyclass typ PlatypusAlgorithm () -> platypus_raw[typ]
end



end # Module
