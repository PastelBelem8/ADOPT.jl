module Platypus


import Base: run

export  CMAES,
        EpsMOEA,
        EvolutionaryStrategy,
        GDE3,
        GeneticAlgorithm,
        IBEA,
        MOEAD,
        NSGAII,
        NSGAIII,
        OMOPSO,
        PAES,
        PESA2,
        SMPSO,
        SPEA2

# Dependencies -----------------------------------------------------------
using PyCall: PyObject, pycall, PyNULL

# ------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------
const platypus = PyNULL()

function __init__()
    copy!(platypus, pyimport_conda("platypus", "platypus-opt", "conda-forge"))
end

# ------------------------------------------------------------------------
# 1. Define Problem
# ------------------------------------------------------------------------
# We need to define the Problem because the Python Algorithm receives a
# Problem object. That must be produced.
#
#  Receives:
#   - nvars
#   - nobjs
#   - nconstrs
#   - function
#   - contraints list of functions
#   - continuous list of lower bounds and upper bounds
#   - integer list of lower bounds and upper bounds
#
# Inherits from Platypus.Problem (has to call super)

abstract type Problem end

struct PlatypusProblem
    pyo::PyObject
end


# ------------------------------------------------------------------------
# 2. Define the Algorithm
# ------------------------------------------------------------------------
# We can export the algorithms methods
abstract type PlatypusAlgorithm end

PyCall.PyObject(x::PlatypusAlgorithm) = x.pyo

macro pyAlgorithm(name)
    quote
        struct $(name) <: PlatypusAlgorithm
            pyo::PyObject
            $(esc(name))(pyo::PyObject) = new(pyo)
            # Create the constructor
            function $(esc(name))(args...; kwargs...)
                new(pycall(platypus[$(QuoteNode(name))], PyObject, args...;kwargs...))
            end
        end
    end
end


@macroexpand @pyAlgorithm CMAES
@pyAlgorithm EpsMOEA
@pyAlgorithm EvolutionaryStrategy
@pyAlgorithm GDE3
@pyAlgorithm GeneticAlgorithm
@pyAlgorithm IBEA
@pyAlgorithm MOEAD
@pyAlgorithm NSGAII
@pyAlgorithm NSGAIII
@pyAlgorithm OMOPSO
@pyAlgorithm PAES
@pyAlgorithm PESA2
@pyAlgorithm SMPSO
@pyAlgorithm SPEA2


# TODO - Platypus Allows to define condition objects (MaxEvaluations and MaxTime)
Base.run(a::PlatypusAlgorithm, condition::Int) = a.pyo[:run](condition)

# FIXME - Update this when solution is wrapped
function results(a::PlatypusAlgorithm)::AbstractMatrix
    sols = a.pyo[:result]
    nsols, ndims = length(sols), a.pyo[:problem][:nobjs]

    res = zeros(Real, ndims, nsols)
    feasible = fill(false, (nsols, 1))

    for (j, sol) in enumerate(sols)
        pyVector = convert(PyVector, sol[:objectives])

        res[:,j] = Vector{Real}(pyvector)
        feasible[j] = sol[:feasible]
    end

    res, feasible
end

function feasibleResults(a::PlatypusAlgorithm)::AbstractMatrix
    results, feasibility = results(a)
    results[feasibility]
end

end # Module
