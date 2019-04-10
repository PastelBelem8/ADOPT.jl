# Imports --------------------------------------------------------------
import Base: show, ==, values

# ----------------------------------------------------------------------
# Auxiliar routines
# ----------------------------------------------------------------------
# Routines to abstract and to the make code more readable/cleaner
"Throws an error if the arguments of a certain type `T` are invalid."
function check_arguments(args...) end

# ---------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------
"""

    AbstractVariable

All variables share the same main behavior, however some discrete
variables require additional fields. For example a variable composed
of specific numbers is considered a discrete variable, yet its values
must be specified in order to be manipulated.

Each variable is either a discrete variable (Int, Set), and, in that case, it might
be comprised of a set of numbers or a range of sequential numbers or it
might be a continuous variable (Real).

See also: [`IntVariable`](@ref), [`RealVariable`](@ref), [`SetVariable`](@ref)
"""
abstract type AbstractVariable end

"Creates and exports variable structure subtype of the `AbstractVariable` type based on the provided fields"
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
        export $(name_sym)
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

# Avoid parameterizing methods
Categorical = Union{Int64, Float64, Number}
CategoricalVector = Union{Vector{Int64}, Vector{Float64}, Vector{Real}}

# Argument Validations
check_arguments(lb::Real, ub::Real, ival::Real) =
    begin
        if lb > ub
            throw(DomainError("lower bound must be less than or equal to the upper bound: $lb ⩽ $ub"))
        elseif lb > ival || ival > ub
            throw(DomainError("the initial value must be within the lower and upper bounds: $lb ⩽ $ival ⩽ $ub"))
        end
    end
check_arguments(lb::Categorical, ub::Categorical, ival::Categorical, values::CategoricalVector) =
    begin
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

# Variable Definitions
@defvariable IntVariable  lower_bound::Int upper_bound::Int initial_value::Int
@defvariable RealVariable lower_bound::Real upper_bound::Real initial_value::Real
@defvariable SetVariable  lower_bound::Categorical upper_bound::Categorical initial_value::Categorical values::CategoricalVector

# Additional Constructors (w/ optional fields)
IntVariable(lbound::Int, ubound::Int) =
    IntVariable(lbound, ubound, floor(Int, (ubound - lbound) / 2) + lbound)
RealVariable(lbound::Real, ubound::Real) =
    RealVariable(lbound, ubound, (ubound - lbound) / 2 + lbound)

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

# Comparators
==(i1::AbstractVariable, i2::AbstractVariable) =
    typeof(i1) == typeof(i2) &&
    lower_bound(i1) == lower_bound(i2) &&
    upper_bound(i1) == upper_bound(i2) &&
    initial_value(i1) == initial_value(i2)
==(i1::SetVariable, i2::SetVariable) =
    invoke(==, Tuple{AbstractVariable, AbstractVariable}, i1, i2) && values(i1) == values(i2)

# Unscalers
unscale(var::T, values::AbstractArray{Y}, old_min, old_max) where{T<:AbstractVariable,Y<:Number} =
        [unscale(var, value, old_min, old_max) for value in values]

unscale(var::RealVariable, value::Real, old_min, old_max) =
    unscale(value, lower_bound(var), upper_bound(var), old_min, old_max)

unscale(var::IntVariable, value::Number, old_min, old_max) =
    round(Int, unscale(value, lower_bound(var), upper_bound(var), old_min, old_max))

unscale(var::SetVariable, value::Number, old_min, old_max) = let
    nval = unscale(value, lower_bound(var), upper_bound(var), old_min, old_max)
    var.values[round(Int64, nval)]
    end

# Export functions
export lower_bound, upper_bound, initial_value, values, ==, unscale

# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------
abstract type AbstractObjective end

# Selectors
func(o::AbstractObjective) = o.func

# Predicates
isAbstractObjective(::T) where{T<:AbstractObjective} = true
isAbstractObjective(::Any) = false

isminimization(o::AbstractObjective, comp) = comp(:MIN, sense(o))

"""
    Objective(λ, n, :MIN)
    Objective(λ, n, :MAX)

The objective is a type that encloses a function, an objective function,
to be used in a optimization problem.

For flexibility and extension purposes the objective also possesses a
coefficiet (or a weight) associated, which can be used to articulate
preferences of multiple objectives. An example of such approach is the
Single Objective weighted sum approach to Multi-Objective problems, where
each objective is assigned a coefficient and then the linear combination of
the multiple objectives is the *function* to be optimized.

# Arguments
- `func::Function`: the function to be computed
- `coefficient::Real`: the weight representing the importance of the objective
function
- `sense::Symbol`: the direction/sense of the objective function, which can
either be to minimize (sense=:MIN) or to maximize (sense=:MAX)
"""
struct Objective <: AbstractObjective
    func::Function
    coefficient::Real
    sense::Symbol

    Objective(f::Function, coefficient::Real=1, sense::Symbol=:MIN) =
        begin   check_arguments(Objective, f, coefficient, sense)
                new(f, coefficient, sense)
        end
end

# Constructor
Objective(f::Function, sense::Symbol) = Objective(f, 1, sense)

# Argument Validations
check_arguments(::Type{Objective}, f::Function, coefficient::Real, sense::Symbol) =
    if !(sense in (:MIN, :MAX))
        throw(DomainError("unrecognized sense $sense. Valid values are {MIN, MAX}"))
    end

# Selectors
coefficient(o::Objective) = o.coefficient
sense(o::Objective) = o.sense

direction(o::Objective) = o.sense == :MIN ? -1 : 1
directions(v::Vector) = [direction(o) for o in v]

# Predicate
isObjective(::Objective)::Bool = true
isObjective(::Any)::Bool = false

isminimization(o::Objective) = invoke(isminimization, Tuple{AbstractObjective, Function}, o, ==)

# Comparators
==(o1::Objective, o2::Objective) =
    func(o1) == func(o2) && coefficient(o1) == coefficient(o2) && sense(o1) == sense(o2)

# Application
nobjectives(::Objective) = 1

"Applies the objective's function to provided arguments"
apply(o::T, args...) where {T <: AbstractObjective} = func(o)(args...)

"Evaluates the true value of the objective"
evaluate(o::Objective, args...) = coefficient(o) .* apply(o, args...)

"""
    SharedObjective(λ, [n1, n2, ...], [:MIN, :MAX, ...])

The shared objective is a that encloses multiple objective functions in a
single function. This avoids evaluating multiple the same function multiple
times, which is extremely useful in the case of simulation-based optimization
problems, where simulations are time-consuming and often simulatenously
produce multiple outputs that can be used as objective functions.


# Arguments
- `nobjectives::Int`: the number of objectives represented by this structure
- `func::Function`: the function to be computed
- `coefficients::Vector{Real}`: the weight representing the importance of the objective
functions
- `senses::Vector{Symbol}`: the senses of the objective functions, which can
either be to minimize (sense=:MIN) or to maximize (sense=:MAX)
"""
struct SharedObjective <: AbstractObjective
    nobjectives::Int
    func::Function
    coefficients::Vector{Real}
    senses::Vector{Symbol}

    SharedObjective(f::Function, coefficients, senses::AbstractVector{Symbol}) =
        begin   check_arguments(SharedObjective, coefficients, senses)
                new(length(coefficients), f, coefficients, senses)
        end
end

# Constructors
SharedObjective(f::Function, nobjs::Int) =
    SharedObjective(f, [1 for _ in 1:objs], [:MIN for _ in 1:nobjs])

SharedObjective(f::Function, coefficients::Vector{T}) where{T<:Real} =
    SharedObjective(f, coefficients, [:MIN for _ in 1:length(coefficients)])

SharedObjective(f::Function, senses::Vector{Symbol}) =
    SharedObjective(f, [1 for _ in 1:length(senses)], senses)

# Argument Validations
check_arguments(::Type{SharedObjective}, coefficients, senses) =
    let valid_sense(sense) = sense in (:MIN, :MAX)
        if length(coefficients) != length(senses)
            throw(DimensionMismatch("`coefficients` and `senses` should have the same dimension"))
        elseif isempty(filter(valid_sense, senses))
            throw(DomainError("unrecognized sense in `senses`. Valid values are {MIN, MAX}"))
        end
    end

# Selectors
nobjectives(o::SharedObjective) = o.nobjectives

coefficient(o::SharedObjective, i::Union{Int,Colon}=(:)) =
    let coeffs = coefficients(o)
        i == (:) || 0 < i < length(coeffs) ? coeffs[i] : throw(BoundsError(coeffs, i))
    end
coefficients(o::SharedObjective) = o.coefficients

sense(o::SharedObjective, i::Union{Int,Colon}=(:)) =
    let snses = senses(o)
        i == (:) || 0 < i < length(snses) ? snses[i] : throw(BoundsError(snses, i))
    end
senses(o::SharedObjective) = o.senses

direction(o::SharedObjective, i::Union{Int,Colon}=(:)) =
    let direction(sense) = sense == :MIN ? -1 : 1
        map(direction, sense(o, i))
    end
directions(v::Vector{SharedObjective}) = [direction(o) for o in v]

# Predicates
isSharedObjective(::SharedObjective) = true
isSharedObjective(::Any) = false

isminimization(o::SharedObjective) =
    invoke(isminimization, Tuple{AbstractObjective, Function}, o, in)

# Application

"Evaluates the true value of the objective"
evaluate(o::SharedObjective, args...) = apply(o, args...) .* coefficients(o)'

# Comparators
==(o1::SharedObjective, o2::SharedObjective) =
    func(o1) == func(o2) && coefficient(o1) == coefficient(o2) &&
    sense(o1) == sense(o2)

==(o1::SharedObjective, o2::Objective) = ==(o1::Objective, o2::SharedObjective) = false

export AbstractObjective, Objective, SharedObjective
export nobjectives, objectives, coefficient, coefficients, sense, senses,
        direction, directions, isminimization, apply, evaluate
# ---------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------
struct Constraint
    func::Function
    coefficient::Real
    operator::Function

    Constraint(f::Function, coefficient::Real=1, operator::Function=(==)) =
        begin
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

# Comparators
==(o1::Constraint, o2::Constraint) =
    func(o1) == func(o2) && coefficient(o1) == coefficient(o2) && operator(o1) == operator(o2)

# Argument Validations
check_arguments(t::Type{Constraint}, f::Function, coefficient::Real, op::Function) =
    if !(Symbol(op) in (:(==), :(!=), :(>=), :(>), :(<=), :(<)))
        throw(DomainError("unrecognized operator $op. Valid operators are {==, !=, =>, >, <=, <}"))
    end

# Application
"Applies the constraint's function to provided arguments"
apply(c::Constraint, args...) = func(c)(args...)

"Evaluates the value of constraint"
evaluate(c::Constraint, args...) = operator(c)(apply(c, args...), 0) ? 0 : apply(c, args...)

"Evaluates the value of the constraint relative to 0"
issatisfied(c::Constraint, c_value) = operator(c)(c_value, 0)

# "Evaluates the magnitude of the constraint violation. It is meant to be used for penalty constraints"
# evaluate_penalty(c::Constraint, args...)::Real =
#     begin
#         if Symbol(operator(c)) == :(!=)
#             throw(MethodError("penalty constraint for symbol $(operator(c)) is not defined"))
#         end
#         issatisfied(c, args...) ? 0 : abs(apply(c, args...)) * coefficient(c)
#     end
# evaluate_penalty(Cs::Vector{Constraint}, args...)::Real =
#     sum([evaluate_penalty(c, args...) for c in Cs])

penalty(cs::Vector{Constraint}, cs_values) = let
    unsatisfied = map((c, cval) -> !issatisfied(c, cval), cs, cs_values)
    cs_unsatisfied = cs[unsatisfied]

    if isempty(cs_unsatisfied) 0 else
        # TODO - There is no default behavior to classify the penalty for != constraints
        if :(!=) in map(operator, cs_unsatisfied)
            throw(MethodError("penalty constraint for symbol $(operator(c)) is not defined"))
        end

        cs_unsatisfied_coeffs = map(coefficient, cs_unsatisfied)
        cs_unsatisfied_values = map(abs, cs_values[unsatisfied])

        sum(cs_unsatisfied_coeffs .* cs_unsatisfied_values)
    end
end

export Constraint
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
- `constraint::Vector{Real}`: The values of the constraints, assigned after the
evaluation of the solution. For each constraint that is satisfied exhibit `true`,
otherwise `false`.
- `constraint_violation::Real=0`: The magnitude of the constraint violation,
assigned after the evaluation of the solution.
- `feasible::Bool=true`: True if the solution does not violate any constraint,
and false otherwise.
- `evaluated::Bool=true`: True if the solution was evaluated, false otherwise.

# Examples
julia> Solution([1,2,3])
Solution(Real[1, 2], Real[], Bool[], 0, true, false)

julia> Solution([])
DomainError
"""
struct Solution
    variables::Vector{Real}
    objectives::Vector{Real}
    constraints::Vector{Real}

    constraint_violation::Real
    feasible::Bool
    evaluated::Bool

    Solution(vars::Vector{T}, objectives::Vector{Y}, constraints::Vector{Z}, constraint_violation::Real, feasible::Bool=true, evaluated::Bool=true) where{T<:Real, Y<:Real, Z<:Real} =
        begin   check_arguments(Solution, vars, objectives, constraints, constraint_violation, feasible, evaluated)
                new(vars, objectives, constraints, constraint_violation, feasible, evaluated)
        end

    Solution(vars::Vector{T}) where{T<:Real} =
        Solution(vars, Vector{Real}(), Vector{Real}(), 0, true, false)

    Solution(vars::Vector{T}, objs::Vector{Y}) where{T<:Real,Y<:Real} =
        Solution(vars, objs, Vector{Real}(), 0, true, true)

    Solution(vars::Vector{T}, constraints::Vector{Y}, constraint_violation::Real, feasible::Bool=true) where{T<:Real, Y<:Real} =
        Solution(vars, Vector{Real}(), constraints, constraint_violation, feasible, false)
end

# Typers
typeof_variables(::Type{Solution}) = Vector{Real}
typeof_objectives(::Type{Solution}) = Vector{Real}
typeof_constraints(::Type{Solution}) = Vector{Real}
typeof_constraint_violation(::Type{Solution}) = Real
typeof_feasible(::Type{Solution}) = Bool
typeof_evaluated(::Type{Solution}) = Bool

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

# Argument Validations
# TODO - CHANGE THIS
check_arguments(::Type{Solution}, vars::Vector{T}, objs::Vector, constrs::Vector{Real},
                constraint_violation::Real, feasible::Bool, evaluated::Bool) where{T<:Real} =
    if length(vars) < 1
        throw(DomainError("invalid number of variables $(length(vars)). A solution must be composed by at least one variable."))
    # elseif constraint_violation != 0 && all(constrs)
    #     throw(DomainError("invalid value for constraint_violation $(constraint_violation). To have constraint violation it is necessary that one of the constraints is not satisfied."))
    end

# -----------------------------------------------------------------------
# Solution Convert Routines FIXME - Not in the best place
# -----------------------------------------------------------------------
import Base: convert
Base.convert(::Type{Solution}, x, y) = Solution(convert(typeof_variables(Solution), x),
                                            convert(typeof_objectives(Solution), y))
Base.convert(::Type{Solution}, x, y, cs::Vector{Constraint}, cs_values) = let
    variables = convert(typeof_variables(Solution), x)
    objectives = convert(typeof_objectives(Solution), y)

    # Constraints
    constraints = convert(typeof_constraints(Solution), cs_values)
    constraint_violation = penalty(cs, cs_values)

    feasible = constraint_violation != 0
    Solution(variables, objectives, constraints, constraint_violation, feasible, true)
end

Base.convert(::Type{Vector{Solution}}, X, y, cs, cs_values) =
    isempty(cs) ?
        map(1:size(X, 2)) do sample
            convert(Solution, X[:, sample], y[:, sample]) end :
        map(1:size(X, 2)) do sample
            convert(Solution, X[:, sample], y[:, sample], cs, cs_values[:, sample]) end

export Solution

# ---------------------------------------------------------------------
# Generic Model / Problem - Interface
# ---------------------------------------------------------------------
abstract type AbstractModel end

# Selectors
constraints(m::AbstractModel) = deepcopy(m.constraints)
objectives(m::AbstractModel) = deepcopy(m.objectives)
variables(m::AbstractModel) = deepcopy(m.variables)

unsafe_objectives(m::T) where{T<:AbstractModel} = m.objectives

nconstraints(m::AbstractModel) = length(m.constraints)
nobjectives(m::AbstractModel) = throw(MethodError("method is not supported"))
nvariables(m::AbstractModel) = length(m.variables)

unscalers(m::AbstractModel, old_min::Int=0, old_max::Int=1) = map(variables(m)) do var
    (val) -> unscale(var, val, old_min, old_max)
    end

# ---------------------------------------------------------------------
# Model / Problem
# ---------------------------------------------------------------------
struct Model <: AbstractModel
    variables::Vector{AbstractVariable}
    objectives::Vector{AbstractObjective}
    constraints::Vector{Constraint}

    Model(nvars::Int, nobjs::Int, nconstrs::Int=0) =
        begin   check_arguments(Model, nvars, nobjs, nconstrs)
                new(Vector{AbstractVariable}(undef, nvars), Vector{AbstractObjective}(undef, nobjs), Vector{Constraint}(undef, nconstrs))
        end
    Model(vars::Vector{T}, objs::Vector{Y}, constrs::Vector{Constraint}=Vector{Constraint}()) where{T<:AbstractVariable, Y<:AbstractObjective} =
        begin   check_arguments(Model, vars, objs, constrs)
                new(vars, objs, constrs)
        end
end
# Selectors
nobjectives(m::Model) =
    try
        sum(map(nobjectives, m.objectives))
    catch y
        if isa(y, UndefRefError)
            length(m.objectives)
        else
            y
        end
    end

aggregate_function(model::Model, transformation=flatten) =
    nconstraints(model) > 0  ? ((x...) -> transformation([apply(o, x...) for o in objectives(model)]),
                                          transformation([apply(c, x...) for c in constraints(model)])) :
                               ((x...) -> transformation([apply(o, x...) for o in objectives(model)]))

# Predicates
isModel(c::AbstractModel)::Bool = true
isModel(c::Any)::Bool = false

ismixedtype(m::AbstractModel)::Bool = length(unique(map(typeof, variables(m)))) > 1

# Argument Validations
check_arguments(::Type{Model}, nvars::Int, nobjs::Int, nconstrs::Int) =
    let err = (x, y, z) -> "invalid number of $x: $y. Number of $x must be greater than $z"

        if nvars < 1
            throw(DomainError(err("variables", nvars, 1)))
        elseif nobjs < 1
            throw(DomainError(err("objectives", nobjs, 1)))
        elseif nconstrs < 0
            throw(DomainError(err("constraints", nconstrs, 0)))
        end
    end
check_arguments(t::Type{Model}, vars::Vector{T}, objs::Vector{Y}, constrs::Vector{Constraint}) where{T<:AbstractVariable, Y<:AbstractObjective} =
    check_arguments(t, length(vars), length(objs), length(constrs))

evaluate(model::Model, s::Solution) = evaluate(model, variables(s))
evaluate(model::Model, Ss::Vector{Solution}) = [evaluate(model, s) for s in Ss]
evaluate(model::Model, vars::Vector, transformation::Function=flatten) =
    nvariables(model) != length(vars) ?
        throw(DimensionMismatch("the number of variables in the model $(nvariables(model)) does not correspond to the number of variables $(length(vars))")) :
        evaluate(vars, objectives(model), constraints(model), transformation)

evaluate(vars::Vector, objs::Vector, cnstrs::Vector, transformation::Function=flatten) = let
    start_time = time();
    objs_values, objs_time = @profile objs (o) -> evaluate(o, vars)
    objs_values = objs_values |> transformation

    if isempty(cnstrs)
        write_result("evaluate", time()-start_time, objs_time, vars, objs_values)
        Solution(vars, objs_values)
    else
        cnstrs_values, cnstrs_time = @profile cnstrs (c) -> evaluate(c, vars)

        # Compute penalty
        cnstrs_penalty = penalty(cnstrs, cnstrs_values)
        feasible = iszero(cnstrs_penalty)

        write_result("evaluate", time()-start_time, cnstrs_time, objs_time, vars,
                    cnstrs_values, cnstrs_penalty, feasible, objs_values)
        Solution(vars, objs_values, cnstrs_values, cnstrs_penalty, feasible, true)
    end
end

export AbstractModel, Model, unscalers, constraints, variables, objectives

# ---------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------
abstract type AbstractSolver end
max_evaluations!(solver::T, evals) where{T<:AbstractSolver} =
    solver.max_evaluations = evals
nondominated_only!(solver::T, nd_only) where{T<:AbstractSolver} =
    solver.nondominated_only = nd_only

"Solves the modeled problem using the given solver"
solve(solver::AbstractSolver, model::Model) = let
    OPTIMIZATION_FILES = "$(results_dir())/$(get_unique_string())"
    with(results_file, "$(OPTIMIZATION_FILES)-results.csv", config_file, "$(OPTIMIZATION_FILES).config") do
        write_config("AbstractSolver::solve", solver, model)

        # Create header - Form:
        # <total_time> <time_cnstr>* <time_objs>+ <var>+ <cnstr>* [<penalty>, <feasible>] <obj>+
        header = ["Total Time(s)"]

        cnstrs = map(i -> "c$i", 1:nconstraints(model))
        if !isempty(cnstrs)
            push!(header, map(c -> "Time_$c(s)", cnstrs)...)
        end

        objs = map(i -> "o$i", 1:nobjectives(model))
        time_objs = map(o -> "Time_$o(s)", objs)
        push!(header, time_objs...)

        vars = map(i -> "var$i", 1:nvariables(model))
        push!(header, vars...)

        if !isempty(cnstrs)
            push!(header, cnstrs...)
            push!(header, ["penalty", "feasible"]...)
        end
        push!(header, objs...)

        # Write header
        write_result("AbstractSolver::solve", header)

        # Solve the problem
        solve_it(solver, model)
    end
end

get_solver(::Type{T}, algorithm::Symbol, params, evals, nd_only) where{T<:AbstractSolver} =
    throw("get_solver not implemented for class")

export AbstractSolver, solve
