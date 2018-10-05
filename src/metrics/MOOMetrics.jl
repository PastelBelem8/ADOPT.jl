module MooMetrics

# Define the type ParetoFront

mutable struct ParetoFront
    values::AbstractMatrix
end

# Private
"""
    addPareto!(v, PF)
    Modifies a Pareto Optimal solution `v` to the ParetoFront `PF`.
"""
addpareto!(v::Vector, PF::ParetoFront) =
    begin
        deletedominated!(v, PF)
        PF.values = [PF.values v]
    end

"""
    isnondominated(v, V) -> bool
    Given a set of vectors `V`, returns whether `v` is a non-dominated or
    non-inferior solution for that set. A vector `v` is said to be non-dominated in
    a set `V` if it is not dominated by any of the vectors in `V`.

    # Examples
    ```jldoctest
    julia> a = [4, 2]; A = [1 3 4; 3 2 1];

    julia> isnondominated(a, A)
    false

    julia> a = []
    true
    ```
"""
isnondominated(v::AbstractVector, V::AbstractMatrix) =
    all([dominates(V[:,j], v) for j in 1:ncols(V)])
isparetoOptimal(v::AbstractVector, P::AbstractMatrix) =
    try:
        isnondominated(v, P)
    catch e
        @warn "An unexpected error was thrown when verifying Pareto
        optimality of the vector $(v): $(e).\n We will proceed by ignoring
        this vector."
        false
    end

"""
    dominates(v0, v1) -> bool
    Returns whether `v0` dominates `v1`. `v0` is said to dominate `v1` if `v1` does
    not improve any of the values of `v0`. In other words, if for every i-th
    dimension of `v0`, the value of i-th dimension of `v1` is greater than or equal
    to the value in `v0`.

    # Examples
    ```jldoctest
    julia> a = [4, 2]; A = [1 3 4; 3 2 1];

    julia> dominates(a, A[:,1])
    false

    julia> dominates(A[:,1], a)
    true

    julia> dominates([1, 3], [4, 2])
    false

    julia> dominates([4, 2], [1, 3])
    false
    ```
"""
dominates(v0::AbstractVector, v1::AbstractVector) =
    all(v0 .<= v1)




# Public
"""
    add!(v, PF)

    Adds the `v` to the set of optimal vectors `PF` if `v` is Pareto Efficient.
    Before adding `v` to `PF`, it is necessary to remove dominated solutions.
    If `v` is not Pareto Efficient, then `PF` will not be changed.
"""
add!(v::Vector, PF::ParetoFront) =
    isparetoOptimal(v, PF) ? addpareto!(v, PF) : PF

"""
    deletedominated!(v, PF)
    Deletes the vectors in `PF` dominated by `v`.
"""
deletedominated!(v::Vector, PF::ParetoFront) =
    begin
        nd_ix = [j for j in 1:j if !dominates(v, PF[:, j])]
        PF.values = Pf.values[:, nd_ix]
    end

# Multi-Objective Optimization Metrics

# Reference Metrics
# Reference metrics require the existence of a True Pareto Front set in order
# to compare the quality of the proposed (the approximation) Pareto Front.

# More information on:
#     - Veldhuizen, D. V. (1999). Multi Objective evolutionary algorithms: Classifications, Analysis, New Innovations. Multi Objective evolutionary algorithms. Air Force Institute of Technology, Wright Patterson, Ohio.
#     - Knowles, J., & Corne, D. (2002). On Metrics for Comparing Nondominated Sets. In Proceedings of the 2002 Congress on Evolutionary Computation, CEC 2002 (pp. 711–716).


# Error Ratio (ER)

"""
    errorRatio(T, A) -> e
    Given the true Pareto Front `T`, returns the proportion of non true Pareto points in A.
"""
errorRatio(T::ParetoFront, A::ParetoFront) =
    error("Metric ER is not implemented yet.")

maxPFError(T::ParetoFront, A::ParetoFront) =
    error("Metric MPFE is not implemented yet.")

"""
"""
generationalDistance(T::ParetoFront, A::ParetoFront) =
    error("Metric GD is not implemented yet.")


overallnondominatedvectorgeneration(T::ParetoFront, A::ParetoFront) =
    error("Metric ONVG is not implemented yet.")




end
