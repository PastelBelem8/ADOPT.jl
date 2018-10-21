# TODO - Define routines to process Callable Structures
struct Callable <: Union{Function, Vector{Function}}
    functions
end

macro deftype(name, parentClass, fields...)
    nameStr = string(name)

    # Assumes that every field has a default value
    fieldNames = map(field->field.args[1], fields)
    fieldTypes = map(field->field.args[2], fields)

    # Make parameters
    params = map(name ->  Expr(:kw, name), fieldNames)
    constructorName = esc(name)
    predicateName = esc(Symbol("is", nameStr, "?"))
    structFields = map((name,typ) -> :($(name) :: $(typ)), fieldNames, fieldTypes)

    quote
        export $(name), $(predicateName)
        struct $(esc(name)) <: $(esc(parentClass))
            $(structFields...)
        end
        $(predicateName)(v::$(name)) = true
        $(predicateName)(v::Any) = false
    end
end

@deftype(Variable,
         Any,
         values::Vector{Union{Real, Integer}},
         lowerBound::Union{Real, Integer},
         upperBound::Union{Real, Integer})

@deftype(Solution,
         Any,
         variables::AbstractVector{Variable},
         objectives::AbstractVector{Float64},
         isEvaluated::Bool,
         isFeasible::Bool)

@deftype(Problem,
         Any,
         nvars::Int,
         nobjs::Int,
         nconstrs::Int,
         func::Callable,
         constraints::Callable,
         continuous::Array{Real, 2},
         integer::Array{Integer, 2})

@deftype(Algorithm,
         Any,
         name::String,
         problem::Problem,
         maxEvals::Int)
