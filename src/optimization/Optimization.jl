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
function checkArguments(t::Type{Any}, args...; kwargs...) end

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

const vartypes_map = Dict(  :INT => Int,
:SET => Real,
:REAL => Real )

getVarType(v::Symbol) = haskey(vartypes_map, v) ? vartypes_map[v] : throw(ArgumentError("unexpected type ($v) does not have a matching Julia type."))
getVarType(v::Type{T}) where T<:VarType = getVarType(Symbol(v))

const sym2vartype_map = Dict( :INT => INT, :SET => SET, :REAL => REAL) # FIXME

# Variables -----------------------------------------------------------
abstract type AbstractVariable end

getbasefields(t::Symbol, typ::Type) = [  esc(:(domain::$(Type{sym2vartype_map[t]}))), #FIXME
                                         esc(:(lower_bound::$typ)),
                                         esc(:(upper_bound::$typ)),
                                         esc(:(initial_value::$typ))]

function checkArguments(d, lb::T, ub::T, ival::T) where {T}
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
                checkArguments($(esc(domain)), $(params_names...))
                new($(esc(domain)), $(params_names...))
            end
        end

        $(predicate_name)(v::AbstractVariable)::Bool = v.domain == $(esc(domain)) ? true : false
        $(predicate_name)(v::Any)::Bool = false
        function Base.show(io::IO, v::AbstractVariable)

        end
    end
end

@defvariable INT
@defvariable SET
@defvariable REAL


# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------

struct Objective
    func::Function
    coefficient::Real
    sense::Symbol

    function Objective(f::Function, coefficient::Real=1, sense::Symbol=:MIN)
        checkArguments(Objective, f, coefficient, sense)
        new(f, coefficient, sense)
    end
end

# Constructor
Objective(f::Function, sense::Symbol) = Objective(f, 1, sense)

# Selectors
coefficient(o::Objective) = o.coefficient
func(o::Objective) = o.func

# Predicates
isObjective(o::Objective)::Bool = true
isObjective(o::Any)::Bool = false

isminimization(o::Objective) = o.sense == :MIN

# Representation
function Base.show(io::IO, o::Objective)
    sense = isminimization(o) ? "minimize" : "maximize"
    print("[Objective]:\nSense:\t\t$sense\nFunction:\t$(o.func)\nCoefficient:\t$(o.coefficient)\n")
end

# Argument Validations
function checkArguments(t::Type{Objective}, f::Function, coefficient::Real, sense::Symbol)
    if !(sense in (:MIN, :MAX))
        throw(ArgumentError("unrecognized sense $sense. Valid values are {MIN, MAX}"))
    end
end

# Evaluators
"Applies the objective's function to provided arguments"
apply(o::Objective, args...) = o.func(args...)

"Evaluates the true value of the objective"
evaluate(o::Objective, args...) = coefficient(o) * apply(o, args...)


# Tests
Objective(identity, 1, :MIN)
Objective(identity, 1, :MAX)
Objective(identity, 1, :X)
Objective(identity, 1)
Objective(identity, :MIN)

o = Objective(identity, 1)

apply(o, 2)
coefficient(o) # Should be 1

evaluate(o) # MethodError
evaluate(o, 2)
