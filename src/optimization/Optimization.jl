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

const sym2vartype_map = Dict( :INT => INT, :SET => SET, :REAL => REAL)

# ---------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------
# All variables share the same main behavior, however some discrete
# variables require an additional field.
#
# Each variable has a domain/type associated. Either a variable is discrete
# and in that case it might be a range of integers or a set of discrete
# (real) numbers, or it might be continuous.

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

        $(predicate_name)(v::AbstractVariable) = v.domain == $(esc(domain)) ? true : false
        $(predicate_name)(v::Any) = false

    end
end

@defvariable INT
@defvariable SET
@defvariable REAL

# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------
