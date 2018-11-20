module Pareto

@inline nrows(v::AbstractMatrix) = size(v, 1)
@inline ncols(v::AbstractMatrix) = size(v, 2)

# Pareto Related Concepts
"""
    weakly_dominates(v0, v1)
    weakly_dominates(v0, V[, function])

Determines if `v0` weakly dominates by `v1`. When considering Pareto efficiency,
    a vector `v0` is said to weakly dominate `v1` if it is possible to improve
    at least one objective of `v1`.

Additionally, it determines `v0`'s Pareto weak relationship in respect to set of vectors `V`. By
specifying filtering functions, such as [`all`](@ref) and [`any`](@ref), it is
possible to determine if `v0` is a Weak Pareto Optimum (WPO) or if it just
dominates another vector in `V`.
"""
weakly_dominates(v0::AbstractVector, v1::AbstractVector) = all(v0 .≤ v1) && any(v0 .< v1)
weakly_dominates(v0::AbstractVector, V::AbstractMatrix, f::Function=identity) =
    f([weakly_dominates(v0, V[:, j]) for j in 1:ncols(V)])

"""
    strongly_dominates(v0, v1)

Determines if `v0` strongly dominates by `v1`. When considering Pareto efficiency,
    a vector `v0` is said to strongly dominate `v1` if it is possible to improve
    all the objectives of `v1`.
"""
strongly_dominates(v0::AbstractVector, v1::AbstractVector) = all(v0 .< v1)

"""
    is_nondominated(v, V)
    is_pareto_optimal(v, V)

Given a set of vectors `V`, returns whether `v` is a non-dominated or
non-inferior solution for that set. A vector `v` is said to be non-dominated in
a set `V` if it is not dominated by any of the vectors in `V`.

# Examples
```jldoctest
julia> a = [4, 2]; A = [1 3 4; 3 2 1];

julia> is_nondominated(a, A)
false

julia> is_nondominated([1, 2], A)
true

julia> is_pareto_optimal([1,1], A)
true

```
"""
is_nondominated(v::AbstractVector, V::AbstractMatrix) = weakly_dominates(v, V, all)
is_pareto_optimal = is_nondominated

# Overriden methods
import Base: push!, isempty

mutable struct ParetoResult
    dominated_variables::AbstractMatrix
    dominated_objectives::AbstractMatrix

    nondominated_variables::AbstractMatrix
    nondominated_objectives::AbstractMatrix

    function ParetoResult(dvars::AbstractMatrix, dobjs::AbstractMatrix,
                 ndvars::AbstractMatrix, ndobjs::AbstractMatrix)
        if !(isempty(dvars) && isempty(ndvars)) && (isempty(dobjs) || isempty(ndobjs))
            throw(DomainError("not possible to create Pareto Result with variables but with no objectives"))
        elseif size(dvars, 1) != size(ndvars, 1)
            throw(DimensionMismatch("dominated and nondominated variables must be of the same size"))
        elseif size(dobjs, 1) != size(ndobjs, 1)
            throw(DimensionMismatch("dominated and nondominated objectives must be of the same size"))
        elseif size(dobjs, 2) != size(dvars, 2)
            throw(DimensionMismatch("the number of points is not the same in `dobjs` and `dvars`"))
        elseif size(ndobjs, 2) != size(ndvars, 2)
            throw(DimensionMismatch("the number of points is not the same in `ndobjs` and `ndvars`"))
        end
        new(dvars, dobjs, ndvars, ndobjs)
    end
end

# Constructors
ParetoResult(dvars::AbstractVector, dobjs, ndvars::AbstractVector, ndobjs) =
    ParetoResult(reshape(dvars, (length(dvars)), 1), dobjs, reshape(ndvars, (length(ndvars)), 1), ndobjs)
ParetoResult(dvars, dobjs::AbstractVector, ndvars, ndobjs::AbstractVector) =
    ParetoResult(dvars, reshape(dobjs, (length(dobjs)), 1), ndvars, reshape(ndobjs, (length(ndobjs)), 1))
ParetoResult(dvars::AbstractVector, dobjs::AbstractVector, ndvars::AbstractVector, ndobjs::AbstractVector) =
    ParetoResult(dvars, reshape(dobjs, (length(dobjs)), 1), ndvars, reshape(ndobjs, (length(ndobjs)), 1))
ParetoResult(vars_dims::Int, objs_dims::Int) =
    if vars_dims > 0 && objs_dims > 0
        ParetoResult(Array{Float64}(undef, vars_dims, 0), Array{Float64}(undef, objs_dims, 0),
                 Array{Float64}(undef, vars_dims, 0), Array{Float64}(undef, objs_dims, 0))
    else
        throw(DomainError("`vars_dims` and `objs_dims` must be positive integers"))
    end

# Selectors
dominated_variables(pd::ParetoResult) = pd.dominated_variables
dominated_objectives(pd::ParetoResult) = pd.dominated_objectives
nondominated_variables(pd::ParetoResult) = pd.nondominated_variables
nondominated_objectives(pd::ParetoResult) = pd.nondominated_objectives

total_nondominated(pd::ParetoResult) = size(nondominated_variables(pd), 2)
total_dominated(pd::ParetoResult) = size(dominated_variables(pd), 2)

# Compound Selectors
variables(pd::ParetoResult) =
    hcat(dominated_variables(pd), nondominated_variables(pd))
objectives(pd::ParetoResult) =
    hcat(dominated_objectives(pd), nondominated_objectives(pd))
ParetoFront(pd::ParetoResult) =
    nondominated_variables(pd), nondominated_objectives(pd)

# Modifiers
remove_nondominated!(pd::ParetoResult, ids::Union{AbstractVector, Int}) = begin
    ids = filter(i -> !(i in ids), 1:total_nondominated(pd));
    pd.nondominated_variables = pd.nondominated_variables[:, ids];
    pd.nondominated_objectives = pd.nondominated_objectives[:, ids];
    pd
end
push_dominated!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector) = begin
    if isempty(vars) && isempty(objs) return end

    pd.dominated_variables = [pd.dominated_variables vars];
    pd.dominated_objectives = [pd.dominated_objectives objs];
end
push_nondominated!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector) = begin
    if isempty(vars) && isempty(objs) return end
    pd.nondominated_variables = [pd.nondominated_variables vars];
    pd.nondominated_objectives = [pd.nondominated_objectives objs];
end

function Base.push!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector, dominance::Function=weakly_dominates)
    nondominated_vars = nondominated_variables(pd);
    nondominated_objs = nondominated_objectives(pd);

    if length(vars) != nrows(nondominated_vars)
        throw(DimensionMismatch("`vars` does not have the same dimension as `ndvars`"))
    elseif length(objs) != nrows(nondominated_objs)
        throw(DimensionMismatch("`objs` does not have the same dimension as `ndobjs`"))
    end

    if is_nondominated(objs, nondominated_objs)
        dominated = dominance(objs, nondominated_objs);
        dominated = filter(i-> dominated[i], 1:length(dominated))

        dominated_vars = nondominated_vars[:, dominated];
        dominated_objs = nondominated_objs[:, dominated];

        # Remove dominated solutions from nondominated
        remove_nondominated!(pd, dominated);

        # Push dominated solutions
        map(dominated) do j
            push_dominated!(pd, dominated_vars[:, j], dominated_objs[:, j]);
        end

        # Push Pareto Optimal Solution
        push_nondominated!(pd, vars, objs);
    else
        push_dominated!(pd, vars, objs);
    end
end

Base.push!(pd::ParetoResult, V::AbstractMatrix) =
     [Base.push!(pd, V[:, j]) for j in 1:size(V, 2)]

Base.isempty(pd::ParetoResult) = all(map(isempty, (variables(pd), objectives(pd))))
end # Module
