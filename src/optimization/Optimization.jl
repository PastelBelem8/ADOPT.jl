# Imports --------------------------------------------------------------
import Base: show, ==, values

"Throws an error if the arguments of a certain type `T` are invalid."
function check_arguments(args...) end

"""
    AbstractVariable

Represents a variable (or unknown) in an optimization problem formulation.

Variables have different properties and behaviors, some of which is
shared across them. This abstract type allows to unify their behaviors
through a unique API.

A variable may be discrete or continuous depending on the values it
takes. A discrete variable spans a finite number of values whereas
a continuous variable spans an infinite number of values.

Currently supported variables are Int and Set (discrete variables) and
also Continuous (continuous variable).

See also: [`IntVariable`](@ref), [`RealVariable`](@ref), [`SetVariable`](@ref)
"""
abstract type AbstractVariable end

"Create `AbstractVariable` subtype with provided `name` and `fields`."
macro variable(name, fields...)
    name_str = string(name)
    name_sym = Symbol(name_str)

    # Fields
    fields_names = map(field -> field.args[1], fields)
    fields_types = map(field -> :($(esc(field.args[2]))), fields)

    struct_fields = map((name, typ) -> :($(name)::$(typ)), fields_names, fields_types)

    # Methods
    constructor_name = esc(name_sym)

    quote
        export $(name_sym)
        struct $(name_sym) <: $(esc(AbstractVariable))
            $(struct_fields...)

            function $(constructor_name)($(struct_fields...))
                check_arguments($(fields_names...))
                new($(fields_names...))
            end
        end
    end
end

# Type Alias
Cat = Union{Int64, Float64, Number}
CatVector = Union{Vector{Int64}, Vector{Float64}, Vector{Real}}

check_arguments(lb::Real, ub::Real, ival::Real) =
    if lb > ub
        throw(DomainError("Invalid bound values: lower_bound ($lb) > upper_bound ($ub)"))
    elseif lb > ival || ival > ub
        throw(DomainError("Invalid initial value: ($ival) ∉ [$lb, $ub]"))
    end
check_arguments(lb::Cat, ub::Cat, ival::Cat, values::CatVector) =
    begin
        if length(values) < 1
            throw(DomainError("No values specified"))
        elseif lb ∉ values
            throw(DomainError("Invalid lower bound value: $lb ∉ $values"))
        elseif ub ∉ values
            throw(DomainError("Invalid upper bound values: $ub ∉ $values"))
        elseif ival ∉ values
            throw(DomainError("Invalid initial value: $ival ∉ $values"))
        end
        invoke(check_arguments, Tuple{Real, Real, Real}, lb, ub, ival)
    end

# Variable Definitions
@variable IntVariable lower_bound::Int upper_bound::Int initial_value::Int
@variable RealVariable lower_bound::Real upper_bound::Real initial_value::Real
@variable SetVariable lower_bound::Cat upper_bound::Cat initial_value::Cat values::CatVector

# Additional Constructors (w/ optional fields)
IntVariable(lb::Int, ub::Int) = IntVariable(lb, ub, floor(Int, (ub - lb) / 2) + lb)
RealVariable(lb::Real, ub::Real) = RealVariable(lb, ub, (ub - lb) / 2 + lb)
SetVariable(ival::Cat, values::CatVector) =
    length(values) > 0 ? SetVariable(minimum(values), maximum(values), ival, values) :
    throw(DomainError("No values specified"))
SetVariable(values::CatVector) =
    length(values) > 0 ? SetVariable(values[1], values) :
    throw(DomainError("No values specified"))

# Selectors
lower_bound(var::AbstractVariable) = var.lower_bound
upper_bound(var::AbstractVariable) = var.upper_bound
initial_value(var::AbstractVariable) = var.initial_value
values(var::AbstractVariable) = throw(MethodError("Undefined"))
values(var::SetVariable) = var.values

# Comparators
==(i1::AbstractVariable, i2::AbstractVariable) =
    typeof(i1) == typeof(i2) &&
    lower_bound(i1) == lower_bound(i2) &&
    upper_bound(i1) == upper_bound(i2) &&
    initial_value(i1) == initial_value(i2)
==(i1::SetVariable, i2::SetVariable) =
    invoke(==, Tuple{AbstractVariable,AbstractVariable}, i1, i2) && values(i1) == values(i2)

# Unscalers
"""
    unscale(v::AbstractVariable, val, omin, omax)

Unscale `val` from `[omin, omax]` to the original variable's scale.
"""
unscale(var, vals, omin, omax) =
    [unscale(var, val, omin, omax) for val in vals]

unscale(var::RealVariable, val::Real, omin::Number, omax::Number) =
    unscale(val, lower_bound(var), upper_bound(var), omin, omax)

unscale(var::IntVariable, val::Number, omin::Number, omax::Number) =
    round(Int, unscale(val, lower_bound(var), upper_bound(var), omin, omax))

unscale(var::SetVariable, val::Number, omin::Number, omax::Number) =
    let vals = values(var),
        uns_val = unscale(val, lower_bound(var), upper_bound(var), omin, omax),
        diff = map((v) -> abs(v - uns_val), vals)
        vals[argmin(diff)]
    end

export lower_bound, upper_bound, initial_value, values, ==, unscale

"""
    AbstractObjective

Represents an objective function in an optimization problem formulation.

While a problem formulation typically comprises different objective
functions, in most practical settings we may (for performance
constraints) resort to a _shared objective_. SharedObjectives differ
from the common `Objective` in the number of outputs returned.
Mathematically, one can think of an `Objective` as a mapping
𝒳 → 𝒴, where 𝒳 ∈ ℝᴰ and 𝒴 ∈ ℝ. Conversely, a `SharedObjective`
represents a mapping 𝒰 → 𝒱, where 𝒰 ∈ ℝᴰ and 𝒱 ∈ ℝ.

# Fields:
- `func::Function`: the actual function(s) to execute.
- `coefficient::Real=1`: the importance weight(s) of the objective.
- `sense::Symbol`: the direction/sense of the function. Supported
    values are {:MIN, ::MAX}.

See also: [`Objective`](@ref), [`SharedObjective`](@ref)
"""
abstract type AbstractObjective end

# Selectors
func(o::AbstractObjective) = o.func

# Predicates
isminimization(s::Symbol) = :MIN == s

# Comparators
==(o1::T, o2::T) where {T <: AbstractObjective} =
    func(o1) == func(o2) && coefficient(o1) == coefficient(o2) &&
    sense(o1) == sense(o2)

# Application
"Apply the objective to provided arguments."
apply(o::T, args...) where {T <: AbstractObjective} = func(o)(args...)

"Compute the true value of the objective."
evaluate(o::T, args...) where {T <: AbstractObjective} =
    coefficient(o) .* apply(o, args...)

"""
    Objective(λ, n, :MIN)
    Objective(λ, n, :MAX)

Define an objective function whose result is unidimensional.

For flexibility and extension purposes the objective also possesses a
coefficient (or a weight) associated, which can be used to articulate
preferences for multiple objectives. This can be useful for
scalarization (_i.e._, a single objective weighted sum approach),
where each objective is assigned a weight and the linear combination of
multiple objectives is the `function` to be optimized.

See Also: [`SharedObjective`](@ref)
"""
struct Objective <: AbstractObjective
    func::Function
    coefficient::Real
    sense::Symbol

    Objective(f::Function, coeff::Real = 1, sense::Symbol = :MIN) = begin
        check_arguments(Objective, f, coeff, sense)
        new(f, coeff, sense)
    end
end

# Constructor
Objective(f::Function, sense::Symbol) = Objective(f, 1, sense)

# Argument Validations
check_arguments(::Type{Objective}, f::Function, coef::Real, sense::Symbol) =
    if !(sense in (:MIN, :MAX))
        throw(DomainError("Invalid optimization goal: $sense ∉ {:MIN, :MAX}"))
    end

# Selectors
coefficient(o::Objective) = o.coefficient
sense(o::Objective) = o.sense

direction(o::Objective) = o.sense == :MIN ? -1 : 1
direction(os::Vector{T}) where {T <: AbstractObjective} = vcat(map(direction, os)...)

isminimization(o::Objective) = isminimization(sense(o))

# Application
nobjectives(::Objective) = 1

"""
    SharedObjective(λ, [n1, n2, ...], [:MIN, :MAX, ...])

Encloses multiple output dimensions in a single function.

In the context of optimization, a shared objective allow us to
simultaneously obtain multiple values for the optimization problem
through a simple function call.  This is an abstraction, proven to
be extremely useful in the case of simulation-based optimization,
where evaluations (or simulations) are time-consuming and often
produce distinct outputs that can be used as objective functions.

Naturally, this abstraction requires additional information when
compared to a simple `Objective`, namely, the total number of
objectives considered in this _aggregate_ objective and their
corresponding coefficients and senses (or directions).

# Fields
- `n::Int8`: the dimensions of the output, i.e., number of objectives.
- `func::Function`: the function to be computed.
- `coefficients::Vector{Number}: the importance weight(s) of the objectives.
    By default each objective is assigned equal importance of 1.
- `senses::Vector{Symbol}`: the direction/sense of the objectives. Supported
    values are {:MIN, ::MAX}. By default, each objective is assumed to be a
    minimization problem.

See Also: [`Objective`](@ref)
"""
struct SharedObjective <: AbstractObjective
    n::Int
    func::Function
    coefficients::Vector{Real}
    senses::Vector{Symbol}

    SharedObjective(f::Function, coefficients, senses::AbstractVector{Symbol}) =
        begin   check_arguments(SharedObjective, coefficients, senses)
                new(length(coefficients), f, coefficients, senses)
        end
end
SharedObjective(f::Function, nobjs::Int) =
    SharedObjective(f, [1 for _ in 1:nobjs], [:MIN for _ in 1:nobjs])
SharedObjective(f::Function, coefficients::Vector{T}) where{T<:Real} =
    SharedObjective(f, coefficients, [:MIN for _ in 1:length(coefficients)])
SharedObjective(f::Function, senses::Vector{Symbol}) =
    SharedObjective(f, [1 for _ in 1:length(senses)], senses)

# Argument Validations
check_arguments(::Type{SharedObjective}, coefficients, senses) =
    let valid_sense(sense) = sense in (:MIN, :MAX)
        if length(coefficients) != length(senses)
            throw(DimensionMismatch("`coefficients` and `senses` differ in size."))
        elseif isempty(filter(valid_sense, senses))
            throw(DomainError("Invalid value in `senses` ∉ {:MIN, :MAX}."))
        elseif length(coefficients) < 2
            throw(DomainError("Invalid definition 1 objective: use `Objective` instead."))
        end
    end

# Selectors
nobjectives(o::SharedObjective) = o.n

coefficient(o::SharedObjective, i=(:)) = coefficients(o)[i]
coefficients(o::SharedObjective) = o.coefficients

sense(o::SharedObjective, i=(:)) = senses(o)[i]
senses(o::SharedObjective) = o.senses

direction(o::SharedObjective, i = (:)) =
    let senses = sense(o, i), get_dir = s -> s == :MIN ? -1 : 1
        if i isa Int
            get_dir(senses)
        else
            # flatten the vector of vectors into a single vector :)
            map(get_dir, senses)
        end
    end

isminimization(o::SharedObjective) = all(map(isminimization, sense(o)))
==(o1::SharedObjective, o2::Objective) = ==(o1::Objective, o2::SharedObjective) = false

evaluate(o::SharedObjective, args...) = sum(coefficient(o) .* collect(apply(o, args...)))

export AbstractObjective, Objective, SharedObjective
export nobjectives, objectives, coefficient, coefficients, sense, senses,
        direction, isminimization, apply, evaluate

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

"Evaluates the value of the constraint relative to 0"
issatisfied(c::Constraint, c_value) = operator(c)(c_value, 0)

evaluate(cs::Vector{Constraint}, cs_values) = let
    unsatisfied = map((c, cval) -> !issatisfied(c, cval), cs, cs_values)
    cs_unsatisfied = cs[unsatisfied]

    if isempty(cs_unsatisfied)
        0
    else
        cs_values_unsatisfied = cs_values[unsatisfied]

        cs_unsatisfied_coeffs = map(coefficient, cs_unsatisfied)
        cs_unsatisfied_values = map(cval -> abs(cval) + abs(ϵ()), cs_values_unsatisfied)

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

# Argument Validations
# TODO - CHANGE THIS
check_arguments(::Type{Solution}, vars::Vector{T}, objs::Vector, constrs::Vector{Real},
                constraint_violation::Real, feasible::Bool, evaluated::Bool) where{T<:Real} =
    if length(vars) < 1
        throw(DomainError("SOLUTION: invalid number of variables $(length(vars)). A solution must be composed by at least one variable."))
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

    feasible = constraint_violation == 0
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
                new(
                  Vector{AbstractVariable}(undef, nvars),
                  Vector{AbstractObjective}(undef, nobjs),
                  Vector{Constraint}(undef, nconstrs))
        end
    Model(
      vars::Vector{T},
      objs::Vector{Y},
      constrs::Vector{Constraint}=Vector{Constraint}()
      ) where{T<:AbstractVariable, Y<:AbstractObjective} = begin
        check_arguments(Model, vars, objs, constrs)
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
check_arguments(
  t::Type{Model},
  vars::Vector{T},
  objs::Vector{Y},
  constrs::Vector{Constraint}) where{T<:AbstractVariable, Y<:AbstractObjective} =
    check_arguments(t, length(vars), length(objs), length(constrs))

evaluate(model::Model, s::Solution) = evaluate(model, variables(s))
evaluate(model::Model, Ss::Vector{Solution}) = [evaluate(model, s) for s in Ss]
evaluate(model::Model, vars::Vector, transformation::Function=flatten) =
    nvariables(model) != length(vars) ?
        throw(DimensionMismatch("the number of variables in the model $(nvariables(model)) does not correspond to the number of variables $(length(vars))")) :
        evaluate(vars, objectives(model), constraints(model), transformation)

evaluate(vars::Vector, objs::Vector, cnstrs::Vector, transformation::Function=flatten) = let
    start_time = time();
    eval_objectives() = let
        objs_values, objs_time = @profile objs (o) -> evaluate(o, vars) # TODO -
        transformation(objs_values), objs_time
    end

    if !isempty(cnstrs)
        println("===== CONSTRAINED ====")
        cnstrs_values, cnstrs_time = @profile cnstrs (c) -> apply(c, vars)

        # Compute penalty
        cnstrs_penalty = penalty(cnstrs, cnstrs_values)
        feasible = iszero(cnstrs_penalty)
        println(">>> Variables: $(vars) \n>>> is_feasible: $(feasible)")

        # TODO - Make this dependent on a  user parameter
        # Problem: Up to now, I am only concerned with hard-constrained optimization
        # problems. This means, I want to minimize the number of unfeasible solutions.
        # In a previous solution, I checked whether the solution was feasible and
        # if so, I would evaluate the objective functions. If not, I would assign
        # the objective-vector 0. However, this is not the solution, as depending
        # whether I'm dealing with a minimization objective, this may be considered
        # optimal, leading the algorithm to focus on that region of the solution space.
        # ----------------------------------------------------------------------------
        if feasible
            objs_values, objs_time = eval_objectives()
        # That being said, the following code decides depending on the objective sense
        # if these values should be as larger or as smaller as possible.
        else
            type_int = typeof(1)
            objs_senses = flatten(map(sense, objs))
            objs_values = map(s -> s == :MIN ? typemax(type_int) : typemin(type_int), objs_senses)
            objs_time = -1 * ones(length(objs_values))
        end

        # Note: To create a more realistic case, whenever we are facing an unfeasible
        write_result("evaluate", time()-start_time, cnstrs_time, objs_time, vars,
                    cnstrs_values, cnstrs_penalty, feasible ? "TRUE" : "FALSE", objs_values)

        Solution(vars, objs_values, cnstrs_values, cnstrs_penalty, feasible, true)
    else
        println("===== UNCONSTRAINED ====")
        objs_values, objs_times = eval_objectives()
        write_result("evaluate", time()-start_time, objs_time, vars, objs_values)
        Solution(vars, objs_values)
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
    create_temp_dir(results_dir())
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
