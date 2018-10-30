# Imports
import Base: convert

# Platypus Solver -------------------------------------------------------

"""
    PlatypusSolver
"""
struct PlatypusSolver <: AbstractSolver
    algorithm_name::Symbol
    max_evaluations::Int
    population_size::Int
    # offspring_size::Int
    # generator::Symbol ??
    # selector::Symbol
    # variator::Symbol
    # comparator::Symbol ??

    PlatypusSolver() = new(:SPEA2, 120, 30)
end

# Converter routines ----------------------------------------------------
# These routines provide the interface between the solver and the
# Platypus library.
# -----------------------------------------------------------------------
platypus_function(objectives, constraints)::Function =
    length(constraints) > 0  ? ((x...) -> return [func(o)(x...) for o in objectives],
                                                 [func(c)(x...) for c in constraints]) :
                               ((x...) -> return [func(o)(x...) for o in objectives])

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
  objs = objectives(m)
  Platypus.set_directions!(problem, directions(objs))
  Platypus.set_function!(problem, platypus_function(objs, constrs))
  problem
end

convert(::Platypus._Constraint, c::Constraint) =
    Platypus._Constraint(string(operator(c)), 0)
convert(::Type{Platypus.PlatypusWrapper}, variable::IntVariable) =
    Platypus._Integer(lower_bound(variable), upper_bound(variable))
convert(::Type{Platypus.PlatypusWrapper}, variable::RealVariable) =
    Platypus._Real(lower_bound(variable), upper_bound(variable))

# Solver routines ------------------------------------------------------
struct PlatypusSolverException <: Exception
  param::Symbol
  value::Any
  reason::Type
  fix::Any
end
Base.showerror(io::IO, e::PlatypusSolverException) =
  print(io, "$(e.param) with value $(e.value) of type $(typeof(e.value)) is $(e.reason).$(e.fix)")

function check_solver(solver::PlatypusSolver)
    if !Platypus.supports(solver.algorithm_name)
        throw(PlatypusSolverException(
                :algorithm_name,
                solver.algorithm_name,
                "is currently not supported by Platypus.",
                "Supported Algorithms: $(Platypus.supported_algorithms())"))
    elseif solver.population_size > solver.max_evaluations
        throw(PlatypusSolverException(
                :max_evaluations,
                solver.max_evaluations,
                "is greater than the population size with value $(solver.population).",
                "Population size should be smaller or equal to max_evaluations."))
    end
    # Check if mandatory arguments are being populated
    # Check if kwargs are present
end

function solve(solver::PlatypusSolver, model::Model)
    check_solver(solver)

    problem = convert(Platypus._Problem, model)
    algorithm = Platypus.Algorithm(solver.algorithm_name, problem)
    solve(algorithm, solver.max_evaluations)
end



#  Test
m = Model([IntVariable(0, 100, 0)],[Objective(identity)])
p = convert(Platypus._Problem, m)
a = Platypus.Algorithm(:SPEA2, p)
Platypus.solve(a, 3)
