# Imports
import Base: convert

# Platypus Solver -------------------------------------------------------

"""
    PlatypusSolver
"""
struct PlatypusSolver <: AbstractSolver
    algorithm_name::Symbol
    algorithm_params::Dict{Symbol,Any}
    max_evaluations::Int
    function PlatypusSolver(algorithm, algorithm_params=Dict{Symbol, Any}(), max_eval=100)
        check_arguments(PlatypusSolver, algorithm_params, max_eval)
        new(algorithm, algorithm_params, max_eval)
    end
end

# get_algorithm(solver::PlatypusSolver) = Platypus.Algorithm()
get_algorithm_name(solver::PlatypusSolver) =   solver.algorithm_name
get_max_evaluations(solver::PlatypusSolver) =  solver.max_evaluations
get_algorithm_params(solver::PlatypusSolver) = solver.algorithm_params
get_algorithm_params(solver::PlatypusSolver, param::Symbol) =
  get(get_algorithm_params(solver), param, nothing)

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

function convert(::Type{Solution}, s::Platypus._Solution)
  variables = Platypus.get_variables(s)
  objectives = Platypus.get_objectives(s)
  # constraints = Platypus.get_constraints(s)
  # constraint_violation = Platypus.get_constraint_violation(s)
  # feasible = Platypus.get_feasible(s)
  # evaluated = Platypus.get_evaluated(s)
  #
  # println(variables)
  # println(objectives)
  # println(constraints)
  # println(constraint_violation)
  # println(feasible)
  # println(evaluated)
  #
  # Solution(variables, objectives, constraints, constraint_violation, feasible, evaluated)
end
convert(::Type{Vector{Solution}}, ss::Vector{Platypus._Solution}) =
  [convert(Solution, s) for s in ss]


# Solver routines ------------------------------------------------------
struct PlatypusSolverException <: Exception
  param::Symbol
  value::Any
  reason::Type
  fix::Any
end
Base.showerror(io::IO, e::PlatypusSolverException) =
  print(io, "$(e.param) with value $(e.value) of type $(typeof(e.value)) is $(e.reason).$(e.fix)")



# Argument Verification
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


function check_model(solver::PlatypusSolver, model::Model)
    # verificar q modelo é mixed type e tem definido o variator
end


function solve(solver::PlatypusSolver, model::Model)
    check_solver(solver)
    check_model(solver, model)

    problem = convert(Platypus._Problem, model)

    a = get_algorithm_name(solver)
    evals = get_max_evaluations(solver)
    extra_params = get_algorithm_params(solver)

    Platypus.solve(algorithm, problem, max_eval=evals, algorithm_params=extra_params)
end


#  Test
m = Model([IntVariable(0, 100, 2), IntVariable(0, 100, 2)],[Objective(x -> x[1] + x[2])])
p = convert(Platypus._Problem, m)
Platypus.get_types(p)
a = Platypus.Algorithm(:SPEA2, p)
convert(Vector{Solution}, Platypus.solve(a, 3))[1]
# var = Platypus.solve(a, 100)
#
# size(var)
ismixedtype(m)
