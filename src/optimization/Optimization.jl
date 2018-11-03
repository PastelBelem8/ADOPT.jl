# Imports --------------------------------------------------------------
import Base: show, ==, values

# ----------------------------------------------------------------------
# Auxiliar routines
# ----------------------------------------------------------------------
# Routines to abstract and to the make code more readable/cleaner
"Throws an error if the arguments of a certain type `T` are invalid."
function check_arguments(args...; kwargs...) end

# ---------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------
# All variables share the same main behavior, however some discrete
# variables require an additional field.
#
# Each variable has a domain/type associated. Either a variable is discrete
# and in that case it might be a range of integers or a set of discrete
# (real) numbers, or it might be continuous.

"""
Variable Types
"""
abstract type  VariableType end
struct INT  <: VariableType end
struct SET  <: VariableType end
struct REAL <: VariableType end

Categorical = Union{Int64, Float64, Number}
CategoricalVector = Union{Vector{Int64}, Vector{Float64}, Vector{Real}}

# Variables -----------------------------------------------------------
abstract type AbstractVariable end

macro defvariable(name, fields...)
    name_str = string(name)
    name_sym = Symbol(name_str)

    # Fields
    fields_names = map(field -> field.args[1], fields)
    fields_types = map(field -> :($(esc(field.args[2]))), fields)

    struct_fields = map((name, typ) -> :($(name)::$(typ)), fields_names, fields_types)

    # Methods
    constructor_name = esc(name_sym)
    predicate_name = esc(Symbol("is", name_str))

    quote
        struct $(name_sym) <: $(esc(AbstractVariable))
            $(struct_fields...)

            function $(constructor_name)($(struct_fields...))
                check_arguments($(fields_names...))
                new($(fields_names...))
            end
        end

        $(predicate_name)(v::$(name_sym))::Bool = true
        $(predicate_name)(v::Any)::Bool = false

    end
end

function check_arguments(lb::Real, ub::Real, ival::Real)
    if lb > ub
        throw(DomainError("lower bound must be less than or equal to the upper bound: $lb ⩽ $ub"))
    elseif lb > ival || ival > ub
        throw(DomainError("the initial value must be within the lower and upper bounds: $lb ⩽ $ival ⩽ $ub"))
    end
end

function check_arguments(lb::Categorical, ub::Categorical, ival::Categorical, values::CategoricalVector)
    if length(values) < 1
        throw(DomainError("invalid variable definition with no values"))
    elseif lb ∉ values
        throw(DomainError("the lower bound with value $lb is not within the specified values: $values"))
    elseif ub ∉ values
        throw(DomainError("the upper bound with value $ub is not within the specified values: $values"))
    elseif ival ∉ values
        throw(DomainError("the initial value with value $ival is not in the specified values: $values"))
    end
    invoke(check_arguments, Tuple{Real, Real, Real}, lb, ub, ival)
end

@defvariable IntVariable  lower_bound::Int upper_bound::Int initial_value::Int
@defvariable RealVariable lower_bound::Real upper_bound::Real initial_value::Real
@defvariable SetVariable  lower_bound::Categorical upper_bound::Categorical initial_value::Categorical values::CategoricalVector

IntVariable(lbound::Int, ubound::Int) = IntVariable(lbound, ubound, floor(Int, (ubound - lbound) / 2) + lbound)
RealVariable(lbound::Real, ubound::Real) = RealVariable(lbound, ubound, (ubound - lbound) / 2 + lbound)

SetVariable(init_value::Categorical, values::CategoricalVector) =
    length(values) > 0 ?
        SetVariable(minimum(values), maximum(values), init_value, values) :
        throw(DomainError("invalid variable definition with no values"))
SetVariable(values::CategoricalVector)=
    length(values) > 0 ? SetVariable(values[1], values) : throw(DomainError("invalid variable definition with no values"))

# Selectors
lower_bound(var::AbstractVariable) = var.lower_bound
upper_bound(var::AbstractVariable) = var.upper_bound
initial_value(var::AbstractVariable) = var.initial_value
values(var::AbstractVariable) = throw(MethodError("Undefined for abstract variables"))
values(var::SetVariable) = var.values

# Equality
==(i1::AbstractVariable, i2::AbstractVariable) =
    typeof(i1) == typeof(i2) &&
    lower_bound(i1) == lower_bound(i2) &&
    upper_bound(i1) == upper_bound(i2) &&
    initial_value(i1) == initial_value(i2)
==(i1::SetVariable, i2::SetVariable) =
    invoke(==, Tuple{AbstractVariable, AbstractVariable}, i1, i2) && values(i1) == values(i2)


# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------
struct Objective
    func::Function
    coefficient::Real
    sense::Symbol

    function Objective(f::Function, coefficient::Real=1, sense::Symbol=:MIN)
        check_arguments(Objective, f, coefficient, sense)
        new(f, coefficient, sense)
    end
end

# Constructor
Objective(f::Function, sense::Symbol) = Objective(f, 1, sense)

# Selectors
coefficient(o::Objective) = o.coefficient
func(o::Objective) = o.func
sense(o::Objective) = o.sense
direction(o::Objective) = o.sense == :MIN ? -1 : 1
directions(v::Vector{Objective}) = [direction(o) for o in v]

# Predicates
isObjective(o::Objective)::Bool = true
isObjective(o::Any)::Bool = false

isminimization(o::Objective) = sense(o) == :MIN

# Representation
# function Base.show(io::IO, o::Objective)
#     sense = isminimization(o) ? "minimize" : "maximize"
#     print("[Objective]:\nSense:\t\t$(sense(o))\nFunction:\t$(func(o))\nCoefficient:\t$(coefficient(o))\n")
# end

# Argument Validations
function check_arguments(t::Type{Objective}, f::Function, coefficient::Real, sense::Symbol)
    if !(sense in (:MIN, :MAX))
        throw(DomainError("unrecognized sense $sense. Valid values are {MIN, MAX}"))
    end
end

# Application
"Applies the objective's function to provided arguments"
apply(o::Objective, args...) = func(o)(args...)

"Evaluates the true value of the objective"
evaluate(o::Objective, args...) = coefficient(o) * apply(o, args...)


# ---------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------

struct Constraint
    func::Function
    coefficient::Real
    operator::Function

    function Constraint(f::Function, coefficient::Real=1, operator::Function=(==))
        check_arguments(Constraint, f, coefficient, operator)
        new(f, coefficient, operator)
    end
end
# Constructor
Constraint(f::Function, operator::Function) = Constraint(f, 1, operator)

# Selectors
coefficient(c::Constraint)::Real = c.coefficient
func(c::Constraint)::Function = c.func
operator(c::Constraint)::Function = c.operator

# Predicates
isConstraint(c::Constraint)::Bool = true
isConstraint(c::Any)::Bool = false

# Representation
# function Base.show(io::IO, c::Constraint)
#     print("[Constraint]:\n  $(coefficient(c)) * $(func(c)) $(Symbol(operator(c))) 0\n")
# end

# Argument Validations
function check_arguments(t::Type{Constraint}, f::Function, coefficient::Real, op::Function)
    if !(Symbol(op) in (:(==), :(!=), :(>=), :(>), :(<=), :(<)))
        throw(DomainError("unrecognized operator $op. Valid operators are {==, !=, =>, >, <=, <}"))
    end
end

# Application
"Applies the constraint's function to provided arguments"
apply(c::Constraint, args...) = func(c)(args...)

"Evaluates the true value of the constraint relative to 0"
evaluate(c::Constraint, args...)::Bool = operator(c)(apply(c, args...), 0)

"Evaluates the magnitude of the constraint violation. It is meant to be used for penalty constraints"
function evaluate_penalty(c::Constraint, args...)::Real
    if Symbol(operator(c)) == :(!=)
        throw(MethodError("penalty constraint for symbol $op is not defined"))
    end
    evaluate(c, args...) ? 0 : abs(apply(c, args...)) * coefficient(c)
end

# ---------------------------------------------------------------------
# Model / Problem
# ---------------------------------------------------------------------
struct Model
    variables::Vector{AbstractVariable}
    objectives::Vector{Objective}
    constraints::Vector{Constraint}

    function Model(nvars::Int, nobjs::Int, nconstrs::Int=0)
        check_arguments(Model, nvars, nobjs, nconstrs)
        new(Vector{AbstractVariable}(undef, nvars), Vector{Objective}(undef, nobjs), Vector{Constraint}(undef, nconstrs))
    end
    function Model(vars::Vector{T}, objs::Vector{Objective},
                    constrs::Vector{Constraint}=Vector{Constraint}()) where {T<:AbstractVariable}
        check_arguments(Model, vars, objs, constrs)
        new(vars, objs, constrs)
    end
end

# Selectors
constraints(m::Model)::Vector{Constraint} = deepcopy(m.constraints)
objectives(m::Model)::Vector{Objective} = deepcopy(m.objectives)
variables(m::Model)::Vector{AbstractVariable} = deepcopy(m.variables)

nconstraints(m::Model) = length(m.constraints)
nobjectives(m::Model) = length(m.objectives)
nvariables(m::Model) = length(m.variables)

# Predicates
isModel(c::Model)::Bool = true
isModel(c::Any)::Bool = false

ismixedtype(m::Model)::Bool = length(unique(map(typeof, variables(m)))) > 1


# Representation
# function Base.show(io::IO, c::Constraint)
#     print("[Constraint]:\n  $(coefficient(c)) * $(func(c)) $(Symbol(operator(c))) 0\n")
# end

# Argument Validations
function check_arguments(t::Type{Model}, nvars::Int, nobjs::Int, nconstrs::Int)
    err = (x, y, z) -> "invalid number of $x: $y. Number of $x must be greater than $z"

    if nvars < 1
        throw(DomainError(err("variables", nvars, 1)))
    elseif nobjs < 1
        throw(DomainError(err("objectives", nobjs, 1)))
    elseif nconstrs < 0
        throw(DomainError(err("constraints", nconstrs, 0)))
    end
end

function check_arguments(t::Type{Model},
                        vars::Vector{T},
                        objs::Vector{Objective},
                        constrs::Vector{Constraint}) where {T<:AbstractVariable}
    check_arguments(t, length(vars), length(objs), length(constrs))
end


# ---------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------
"""
    Solution(v)

Creates a candidate solution with values `v`.

# Arguments
- `variables::Vector{Real}`: The values of thes variables
- `objectives::Vector{Real}`: The values of the objectives, assigned after the
evaluation of the solution.
- `constraint::Vector{Bool}`: The values of the constraints, assigned after the
evaluation of the solution.
- `constraint_violation::Real=0`: The magnitude of the constraint violation,
assigned after the evaluation of the solution.
- `feasible::Bool=true`: True if the solution does not violate any constraint,
and false otherwise.
- `evaluated::Bool=false`: True if the solution was evaluated, false otherwise.

# Examples
julia> Solution([1,2,3])
Solution(Real[1, 2], Real[], Bool[], 0, true, false)

julia> Solution([])
Solution(Real[1, 2], Real[], Bool[], 0, true, false)
"""
struct Solution
    variables::Vector{Any} # TODO FIX
    objectives::Vector{Real}
    constraints::Vector{Bool}

    constraint_violation::Real
    feasible::Bool
    evaluated::Bool

    function Solution(v::Vector{T}) where {T<:Real}
        check_arguments(Solution, v)
        new(v, Vector{Real}(), Vector{Bool}(), 0, true, false)
     end
    function Solution(v, objectives, constraints, constraint_violation, feasible, evaluated)
        new(v, objectives, constraints, constraint_violation, feasible, evaluated)
    end
end

# Selectors
variables(s::Solution) = s.variables
objectives(s::Solution) = s.objectives
constraints(s::Solution) = s.constraints
constraint_violation(s::Solution) = s.constraint_violation

nvariables(s::Solution) = length(s.variables)
nobjectives(s::Solution) = length(s.objectives)
nconstraints(s::Solution) = length(s.constraints)

# Predicates
isfeasible(s::Solution) = s.feasible
isevaluated(s::Solution) = s.evaluated

isSolution(s::Solution)::Bool = true
isSolution(s::Any)::Bool = false

# Representation
# function Base.show(io::IO, c::Solution)
#
# end

# Argument Validations
function check_arguments(t::Type{Solution}, vars::Vector{T}) where {T<:Real}
    if length(vars) < 1
        throw(ArgumentError("invalid number of variables $(length(vars)). A solution must be composed by at least one variable."))
    end
end

# ---------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------
abstract type AbstractSolver end

"""
    SolverFactory(s)

Returns the [`Solver`](@ref) associated with `s` if it exists, else it throws
an exception.
"""
SolverFactory(solver::Symbol)::T where {T<:AbstractSolver} =
    haskey(solvers, solver) ? solvers[solver] : throw(ArgumentError("invalid solver $solver was specified"))
SolverFactory(solver::AbstractString)::T where {T<:AbstractSolver} =
    SolverFactory(Symbol(solver))

"Solves the modeled problem using the given solver"
function solve(solver::AbstractSolver, model::Model) end
