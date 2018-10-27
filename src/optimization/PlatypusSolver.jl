# Imports
import Base: convert

# Global Constants
# -------------------------------------------------------------------


const singleobjective_solvers = Dict(
    :GA => PlatypusSolver,
    :sES => PlatypusSolver
)

const multiobjective_solvers = Dict(
    :CMAES => PlatypusSolver,
    :EpsMOEA => PlatypusSolver,
    :EpsNSGAII => PlatypusSolver,
    :GDE3 => PlatypusSolver,
    :IBEA => PlatypusSolver,
    :MOEAD => PlatypusSolver,
    :NSGAII => PlatypusSolver,
    :NSGAIII => PlatypusSolver,
    :PAES => PlatypusSolver,
    :PESA2 => PlatypusSolver,
    :OMOPSO => PlatypusSolver,
    :SMPSO => PlatypusSolver,
    :SPEA2 => PlatypusSolver,
)


# Platypus Solver ---------------------------------------------------

"""
    PlatypusSolver
"""
struct PlatypusSolver <: AbstractSolver
    algorithm_name::Symbol
    max_evaluations::Int
    population_size::Int
    offspring_size::Int
    # generator::Symbol ??
    selector::Symbol
    variator::Symbol
    # comparator::Symbol ??

end

PlatypusSolver()

solve(solver::Type{PlatypusSolver}, model::Model, algorithm::Symbol)

# TODO -
convert(t::Platypus._Problem, x::Model)
