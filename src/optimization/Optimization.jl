# Imports --------------------------------------------------------------
import Base: show

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------
# Routines to abstract and to the make code more readable/cleaner

# Fields
parsefield(v::Expr, _=nothing) = :($(v.args[1])::$(v.args[2]))
parsefield(v::Symbol, typ=nothing) = :($v::$typ)

"Return base fields that comprise a data structure"
function getbasefields(t::Type{Any}, typ::Type) end

"Throws an error if the arguments of a certain type `T` are invalid."
function check_arguments(t::Type{Any}, args...; kwargs...) end

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
abstract type  VarType end
struct INT  <: VarType end
struct SET  <: VarType end
struct REAL <: VarType end

const vartypes_map = Dict(  :INT => Int,  :SET => Real, :REAL => Real )

getVarType(v::Symbol) = haskey(vartypes_map, v) ? vartypes_map[v] : throw(ArgumentError("unexpected type ($v) does not have a matching Julia type."))
getVarType(v::Type{T}) where T<:VarType = getVarType(Symbol(v))

const sym2vartype_map = Dict( :INT => INT, :SET => SET, :REAL => REAL) # FIXME

# Variables -----------------------------------------------------------
abstract type AbstractVariable end

getbasefields(t::Symbol, typ::Type) = [  esc(:(domain::$(Type{sym2vartype_map[t]}))), #FIXME
                                         esc(:(lower_bound::$typ)),
                                         esc(:(upper_bound::$typ)),
                                         esc(:(initial_value::$typ))]

function check_arguments(d::Type{T}, lb, ub, ival) where {T <: VarType}
    if !(typeof(lb) <: getVarType(d))
        throw(TypeError("variable's domain type $d is not compliant with $T"))
    elseif lb > ub
        throw(ArgumentError("lower bound must be less than or equal to the upper bound: $lb ⩽ $ub"))
    elseif lb > ival || ival > ub
        throw(ArgumentError("the initial value must be within the lower and upper bounds: $lb ⩽ $ival ⩽ $ub"))
    end
end

macro defvariable(domain::Symbol, optional_fields...)
    name_str = domain |> string |> titlecase |> x -> "$(x)Variable"
    name_sym = Symbol(name_str)
    varType = getVarType(domain)

    # Fields
    base_fields = getbasefields(domain, varType)
    optional_fields = map(field -> parsefield(field, varType), optional_fields)

    # Make params
    params = base_fields[2:end]
    kw_params = map(field ->  Expr(:kw, field), optional_fields)
    params_names = map(field -> esc(field.args[1].args[1]), params)
    params_names = !isempty(kw_params) ? vcat(params_name, map(field -> field.args[1], optional_fields)) : params_names

    # Methods
    constructor_name = esc(name_sym)
    predicate_name = esc(Symbol("is", name_str))

    quote
        struct $(name_sym) <: AbstractVariable
            $(base_fields...)
            $(optional_fields...)

            function $(constructor_name)($(params...);$(kw_params...))
                check_arguments($(esc(domain)), $(params_names...))
                new($(esc(domain)), $(params_names...))
            end
        end

        $(predicate_name)(v::AbstractVariable)::Bool = v.domain == $(esc(domain)) ? true : false
        $(predicate_name)(v::Any)::Bool = false

    end
end

@defvariable INT
# @defvariable SET values=[1, 3, 7, 9, 11]
@defvariable REAL


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
        throw(ArgumentError("unrecognized sense $sense. Valid values are {MIN, MAX}"))
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
        throw(ArgumentError("unrecognized operator $op. Valid operators are {==, !=, =>, >, <=, <}"))
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

# Representation
# function Base.show(io::IO, c::Constraint)
#     print("[Constraint]:\n  $(coefficient(c)) * $(func(c)) $(Symbol(operator(c))) 0\n")
# end

# Argument Validations
function check_arguments(t::Type{Model}, nvars::Int, nobjs::Int, nconstrs::Int)
    err = (x, y, z) -> "invalid number of $x: $y. Number of $x must be greater than $z"

    if nvars < 1
        throw(ArgumentError(err("variables", nvars, 1)))
    elseif nobjs < 1
        throw(ArgumentError(err("objectives", nobjs, 1)))
    elseif nconstrs < 0
        throw(ArgumentError(err("constraints", nconstrs, 0)))
    end
end

function check_arguments(t::Type{Model},
                        vars::Vector{AbstractVariable},
                        objs::Vector{Objective},
                        constrs::Vector{Constraint})
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
    variables::Vector{Real}
    objectives::Vector{Real}
    constraints::Vector{Bool}

    constraint_violation::Real
    feasible::Bool
    evaluated::Bool

    function Solution(v::Vector{T}) where {T<:Real}
        check_arguments(Solution, v)
        new(v, Vector{Real}(), Vector{Bool}(), 0, true, false)
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
