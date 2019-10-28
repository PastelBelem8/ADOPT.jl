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
weakly_dominates(v0::AbstractVector, v1::AbstractVector, signs=nothing) =
let v0 = (isnothing(v0) && v0) || v0 .* signs,
    v1 = (isnothing(v1) && v1) || v1 .* signs

    isempty(v1) ? true : all(v0 .≤ v1) && any(v0 .< v1)
end
weakly_dominates(v0::AbstractVector, V::AbstractMatrix, signs=nothing, f::Function=identity) =
    isempty(V) ? true : f([weakly_dominates(v0, V[:, j], signs) for j in 1:ncols(V)])
weakly_dominates(V::AbstractMatrix, v0::AbstractVector, signs=nothing, f::Function=identity) =
    isempty(V) ? false : f([weakly_dominates(V[:, j], v0, signs) for j in 1:ncols(V)])

"""
    strongly_dominates(v0, v1)

Determines if `v0` strongly dominates by `v1`. When considering Pareto efficiency,
    a vector `v0` is said to strongly dominate `v1` if it is possible to improve
    all the objectives of `v1`.
"""
strongly_dominates(v0::AbstractVector, v1::AbstractVector, signs=nothing) =
let v0 = (isnothing(v0) && v0) || v0 .* signs,
    v1 = (isnothing(v1) && v1) || v1 .* signs

    all((v0 .* signs) .< (v1 .* signs))
end

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
is_nondominated(v::AbstractVector, V::AbstractMatrix, signs=nothing) = !weakly_dominates(V, v, signs, any)
is_pareto_optimal = is_nondominated

# Overriden methods
import Base: push!, isempty

mutable struct ParetoResult
    dominated_variables::AbstractMatrix
    dominated_objectives::AbstractMatrix

    nondominated_variables::AbstractMatrix
    nondominated_objectives::AbstractMatrix

    signs::Vector

    function ParetoResult(dvars::AbstractMatrix, dobjs::AbstractMatrix,
                 ndvars::AbstractMatrix, ndobjs::AbstractMatrix, senses::Vector{Symbol}=nothing)
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

        signs = isnothing(senses) ? ones(nobjs) : map(s -> s in (:MIN, :MINIMIZE) ? 1 : -1, senses)
        new(dvars, dobjs, ndvars, ndobjs, signs)
    end
end

# Constructors
ParetoResult(dvars::AbstractVector, dobjs, ndvars::AbstractVector, ndobjs, senses=nothing) =
    ParetoResult(reshape(dvars, (length(dvars)), 1), dobjs, reshape(ndvars, (length(ndvars)), 1), ndobjs, is_nd, senses)
ParetoResult(dvars, dobjs::AbstractVector, ndvars, ndobjs::AbstractVector, senses=nothing) =
    ParetoResult(dvars, reshape(dobjs, (length(dobjs)), 1), ndvars, reshape(ndobjs, (length(ndobjs)), 1), senses)
ParetoResult(dvars::AbstractVector, dobjs::AbstractVector, ndvars::AbstractVector, ndobjs::AbstractVector, senses=nothing) =
    ParetoResult(dvars, reshape(dobjs, (length(dobjs)), 1), ndvars, reshape(ndobjs, (length(ndobjs)), 1), senses)
ParetoResult(vars_dims::Int, objs_dims::Int, senses=nothing) =
    vars_dims > 0 && objs_dims > 0 ?
        ParetoResult(Array{Float64}(undef, vars_dims, 0),
                     Array{Float64}(undef, objs_dims, 0),
                     Array{Float64}(undef, vars_dims, 0),
                     Array{Float64}(undef, objs_dims, 0),
                     senses) :
        throw(DomainError("`vars_dims` and `objs_dims` must be positive integers"))

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

# Predicates
is_in(x0, X) = isempty(X) ? false : any(mapslices(x -> x == x0, X, dims=1))

# Modifiers
remove_nondominated!(pd::ParetoResult, ids::Union{AbstractVector, Int}) = begin
    ids = filter(i -> !(i in ids), 1:total_nondominated(pd));
    pd.nondominated_variables = pd.nondominated_variables[:, ids];
    pd.nondominated_objectives = pd.nondominated_objectives[:, ids];
    pd
end
push_dominated!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector) = begin
    if isempty(vars) && isempty(objs) return end
    if is_in(objs, pd.dominated_objectives) return end

    pd.dominated_variables = [pd.dominated_variables vars];
    pd.dominated_objectives = [pd.dominated_objectives objs];
end
push_nondominated!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector) = begin
    if isempty(vars) && isempty(objs) return end
    if is_in(objs, pd.nondominated_objectives) return end

    pd.nondominated_variables = [pd.nondominated_variables vars];
    pd.nondominated_objectives = [pd.nondominated_objectives objs];
end

Base.push!(pd::ParetoResult, vars::AbstractVector, objs::AbstractVector, is_nondominated::Function=is_nondominated, dominance::Function=weakly_dominates) =
    let nondominated_vars = nondominated_variables(pd)
        nondominated_objs = nondominated_objectives(pd)
        signs = pd.signs

        if length(vars) != nrows(nondominated_vars)
            throw(DimensionMismatch("`vars` does not have the same dimension as `ndvars`: $(length(vars))!= $(nrows(nondominated_vars))"))
        elseif length(objs) != nrows(nondominated_objs)
            throw(DimensionMismatch("`objs` does not have the same dimension as `ndobjs`: $(length(objs))!= $(nrows(nondominated_objs))"))
        end

        if is_nondominated(objs, nondominated_objs, signs)
            dominated = dominance(objs, nondominated_objs, signs);
            dominated = filter(i-> dominated[i], 1:length(dominated))

            if !isempty(dominated) && !isempty(nondominated_objs)
                # Push dominated solutions
                map(dominated) do j
                    push_dominated!(pd, nondominated_vars[:, j], nondominated_objs[:, j]);
                end

                # Remove dominated solutions from nondominated
                remove_nondominated!(pd, dominated);
            end
            # Push Pareto Optimal Solution
            push_nondominated!(pd, vars, objs);
        else
            push_dominated!(pd, vars, objs);
        end
    end

Base.push!(pd::ParetoResult, V0::AbstractMatrix, V1::AbstractMatrix) =
     [Base.push!(pd, V0[:, j], V1[:, j]) for j in 1:size(V1, 2)]

Base.isempty(pd::ParetoResult) = all(map(isempty, (variables(pd), objectives(pd))))

end # Module
