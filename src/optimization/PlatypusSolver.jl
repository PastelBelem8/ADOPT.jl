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


# Platypus Solver -------------------------------------------------------

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

# Converter routines ----------------------------------------------------
# These routines provide the interface between the solver and the
# Platypus library.
# -----------------------------------------------------------------------

function convert(::Type{Platypus._Problem}, m::Model)
  # 1. Create Base Problem
  nvars, nobjs, nconstrs = nvariables(m), nobjectives(m), nconstraints(m)
  problem = Platypus._Problem(nvars, nobjs, nconstrs)

  # 2. Refine the Problem instance
  # 2.1. Convert types
  Platypus.set_types!(problem, [convert(Platypus.PlatypusWrapper, v)
                                        for v in variables(m)])

  # 2.2. Convert Constraints
  constrs = []
  if nconstrs > 0
    constrs = constraints(m)
    Platypus.set_constraints!(problem, [convert(_Constraint, c)
                                                for c in constrs])
  end

  # 2.3. Convert Objective Function
  objectives = objectives(m)
  Platypus.set_directions!(problem, directions(objectives))
  Platypus.set_function!(problem, (x...) ->
                                ( [func(o)(x...) for o in objectives],
                                  [func(c)(x...) for c in constrs] ))
  problem
end

convert(::_Constraint, c::Constraint) = _Constraint(string(operator(c)), 0)
convert(::Type{PlatypusWrapper}, variable::IntVariable) =
    _Integer(lower_bound(variable), upper_bound(variable))
convert(::Type{PlatypusWrapper}, variable::RealVariable) =
    _Real(lower_bound(variable), upper_bound(variable))


# Solver routines ------------------------------------------------------

solve(solver::Type{PlatypusSolver}, model::Model)



#  Test
m = Model([IntVariable(0, 3, 3)],[Objective(identity)])
p = convert(_Problem, m)
