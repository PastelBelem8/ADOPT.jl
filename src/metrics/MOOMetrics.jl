module MOOMetrics

# Define the type ParetoFront

mutable struct ParetoFront
    values::AbstractMatrix
end

# Private ----------------------------------------------------------------
"""
    addPareto!(v, PF)
    Modifies a Pareto Optimal solution `v` to the ParetoFront `PF`.
"""
addpareto!(v::Vector, PF::ParetoFront) =
    begin
        deletedominated!(v, PF)
        PF.values = order([v PF.values])
    end

assertDimensions(a::AbstractVector, A::ParetoFront) =
    if length(a) != nrows(A)
        throw(DimensionMismatch("Different objective-space dimensions:
                                    $(length(a)) != $(nrows(A))."))
    end
contains(P::ParetoFront, v::AbstractVector) = contains(P.values, v)
contains(V::AbstractMatrix, v::AbstractVector) =
    any([v == V[:,j] for j in 1:ncols(V)])

order(V::AbstractMatrix) =
    begin
        let indices = Vector(1:ncols(V))
            lt_f = (i0, i1) -> V[:, i0] < V[:, i1]
            sort!(indices,lt=lt_f)
            V[:,indices]
        end
    end

@inline ndim(P::ParetoFront) = nrows(P.values)
@inline nsols(P::ParetoFront) = ncols(P.values)
@inline solution_obj(P::ParetoFront, i::Int, j::Int) = P.values[i,j]
@inline solution(P::ParetoFront, j::Int) = P.values[:,j]
@inline solutions(P::ParetoFront, js::AbstractMatrix{Int}) = P.values[:,j]
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
    all([weaklyDominates(V[:,j], v) for j in 1:ncols(V)])
isparetoOptimal(v::AbstractVector, P::AbstractMatrix) =
    try:
        isnondominated(v, P)
    catch e
        @warn "An unexpected error was thrown when verifying Pareto
        optimality of the vector $(v): $(e).\n We will proceed by ignoring
        this vector."
        false
    end

"Returns whether there is a v1 belonging to `V`, that is weakly dominated by v."
weaklyDominatesAny(v::AbstractVector, V::AbstractArray) =
    any([weaklyDominates(v, V[:, j]) for j in 1:ncols(V)])

"Returns whether `v0` is never worse than v1 and has an objective value that is
at least better than v1."
weaklyDominates(v0::AbstractVector, v1::AbstractVector) = v0 <= v1 && v0 < v1


# Public -----------------------------------------------------------------
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
        nd_ix = [j for j in 1:j if !weaklyDominates(v, solution(PF, j))]
        PF.values = solutions([:, nd_ix])
    end
Base.show(io::IO, pf::ParetoFront) =
    print(io, [solution(pf, j) for j in 1:nsols(pf)]

# ------------------------------------------------------------------------
# Multi-Objective Optimization Metrics
# ------------------------------------------------------------------------

# Depedencies ------------------------------------------------------------
using Distances
using Statistics

# ------------------------------------------------------------------------
# Independent Metrics
# ------------------------------------------------------------------------
"""
    hypervolumeIndicator(r, A) -> s

    Returns the hypervolume of the multi-dimensional region enclosed by `A` and a
    reference point `r`, i.e., computes the size of the region dominated by `A`
    [1]. It is an independent metric, that induces a complete ordering, and it
    is non-cardinal. Different values of `r` might influence differently the
    scores [2, 5]. By default the reference point will be determined based on
    the worst known value in each objective and shift it by a small amount. A
    set with larger hypervolume is likely to present a better set of trade-offs
    than sets with lower hypervolume.

    Moreover, also known as S-metric or Lebesgue measure, the hypervolume
    indicator (HV or HVI) is classified as NP-hard in the number of objectives,
    evidencing the complexity of the metric (unless P=NP, no polynomial
    algorithm for HV exists).

    We provide a current state-of-the art implementation based on Badstreet's
    Phd Thesis [6]. Motivated by the bad performance of currently available
    algorithms, as well as the high complexity of HV, Badstreet proposed two
    algorithms aimed at improving the feasibility of HV in multi-objective
    optimisation:
        (1) IHSO (Incremental Hypervolume by Slicing Objectives)
        (2) IIHSO (Iterated IHSO)
    Due to the high computational impact, this metric is often infeasible for
    problems with many objectives or for problems with large data sets.
"""
hypervolumeIndicator(r::AbstractVector, A::ParetoFront) =
    begin
        assertDimensions(r, A)
        let hvol = 0
            for j in 1:nsols(A)
                # Compute Hypervolume
                hvol += abs(solution_obj(A, 1, j) - r[1]) *
                        abs(solution_obj(A, 2, j) - r[2])
                # Update reference point to prevent overlapping volumes.
                r[2] = solution_obj(A,2, j) # FIXME - It won't probably work for 3D+ (must generalize)
            end
        end
    end


"""
    onvg(A) -> s
    overallNDvectorgeneration(A) -> s

    Returns the cardinality of the approximation set `A`.
    Overall Nondominated Vector Generation (ONVG) is a cardinal measure that
    uses the cardinality of the approximation sets to induce a complete
    ordering on the sets of approxmation, hence being scale independent.

    While being very easy to compute, it is very easy to mis-classify
    approximation sets using exclusively this metric (e.g. approximation sets
    with many points will always be better than sets with fewer points that
    dominate all the points of the first metric). For more information, refer
    to [3].

    Optionally, if the true Pareto Front `T` is known, you might opt to
    compute the ONVG Ratio (ONVGR)[3,4][`overallNDvectorgenerationRatio`](@ref).
"""
overallNDvectorgeneration(A::ParetoFront) = nsols(A)

"Computes the ratio of solutions between the approximation and true sets."
overallNDvectorgenerationRatio(T::ParetoFront, A::ParetoFront) =
    nsols(A) / nsols(T)

onvg(A) = overallNDvectorgeneration(A)
onvgr(A) = overallNDvectorgenerationRatio(A)

"""
    spacing(A) -> s
    spacing(A, b) -> s

    Given an approximation set, computes the variance of the distances between
    each point and its closest neighbor. The Set Spacing (SS) [5] is a measure
    of the uniformity of the distribution of solutions in the approximation set.
    A value of zero would mean that all members of e are equally spaced from
    one another.

    By default, it uses the the cityblock
    distance, also called
    [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) as
    defined by Schott [5]. However, it is also possible to use the Euclidean
    distance as proposed by Deb. et al[4].

    While inducing complete ordering and being based on the cardinality of `A`,
    this metric has low computational overhead. The original proposed
    non-normalized version of the metric [5] might be problematic as the
    distances are not properly normalized. Consider the normalized version
    spacing as proposed in [4] by specifying ```norm=true```.
"""
spacing(A::ParetoFront, norm::bool=false) =
    let nsols = nsols(A)
        min_ds = [minimum_distance(solution(A, j), solution(A, 1:end .!=j),
                                    norm ? Distances.euclidean : Distances.cityblock)
                        for j in 1:nsols]
        Statistics.var(min_ds)
    end


# ------------------------------------------------------------------------
# Reference Metrics
# Reference metrics require the existence of a True Pareto Front set in order
# to compare the quality of the proposed (the approximation) Pareto Front.
# ------------------------------------------------------------------------

# Error Ratio (ER)
"""
    errorRatio(T, A) -> r

    Given a reference set `T` representing the true Pareto Front, returns the
    proportion of non true Pareto points in A. The lower the value of Error
    Ratio (ER) the better the non-dominated set. It induces order and accounts
    for cardinality.
"""
errorRatio(T::ParetoFront, A::ParetoFront) =
    let n = nsols(A)
        errors = [1 for j in 1:n if !contains(T, solution(A, j))]
        sum(errors) / n
    end

"""
    maxPFError(T, A) -> e
    Given a reference set `T` representing the true Pareto Front, determines a
    maximum error band `e` which is the largest minimum distance between each
    vector in `A` and the corresponding closest vector in `T`. This metric is
    called Maximum Pareto Front Error (MPFE) and it was first introduced by
    Veldhuizen [3] in 1999. It is a reference metric that induces ordering and
    it is a non-cardinal metric.

    The lowest the value of `e` the better the approximation set.
"""
maxPFError(T::ParetoFront, A::ParetoFront) =
    let min_ds = [minimum_distance(solution(A, j), T, Distances.euclidean)
                    for j in 1:nsols(A)]
        maximum(min_ds)
    end

minimum_distance(v::AbstractVector, V::AbstractMatrix,
                    metric::Function=Distances.euclidean) =
    minimum(distances(v, V, metric))

@inline distances(v::AbstractVector, V::AbstractMatrix, metric::Function) =
    [metric(solution(V,j), v) for j in 1:ncols(V)]

"""
    generationalDistance(T, A) -> s

    Given a reference set `T` representing the true Pareto Front, determines the
    general progress of `A` towards `T` with smaller values representing better
    progress. It induces ordering (lower scores represent better approximation
    sets) and it does not consider the cardinality of the sets.

    The generational distance (GD) score favours sets with one vector close to
    `T` over a set containing that vector plus others, as long as the others
    are not closer on average than the first one [2]. Unlike the
    [`hypervolumeIndicator`](@ref), GD is very cheap to compute.
"""
# TODO - Complement description of GD. There are several disadvantages
#        associated to the GD metric that should be accounted for. [2], [3]
generationalDistance(T::ParetoFront, A::ParetoFront) =
    let nsols = nsols(A)
        squared_min_dists = [minimum_distance(solution(A, j), T, Distances.euclidean)^2
                                for j in 1:nsols]
        √_sum = √(sum(squared_min_dists))
        √_sum / nsols
    end


"""
    d1(T, A) -> s

    Measures the mean distance, over the points in the reference set `T`, of
    the nearest point in an approximation set `A`.

    Can be used to measure diversity and convergence.
"""


d1r(T::ParetoFront, A::ParetoFront) =
    let nsols = nsols(T)
        Λ = objs_range(T)
        d = (a, t) -> maximum((t - a) * Λ)
        min_dists = [minimum_distance(solution(T, j), solutions(A, 1:end), d)
                        for j in 1:nsols(T)]
        sum(min_dists) / nsols
    end

objs_range(P::ParetoFront) =
    let solutions = solutions(P, 1:end)
        ranges = abs(maximum(solutions, dims = 2) - minimum(solutions, dims=2))
        Λ = 1 ./ ranges
    end


# ------------------------------------------------------------------------
# Direct Comparative Metrics
# ------------------------------------------------------------------------
"""
    coverage(A, B) -> s

Returns the fraction of solutions in `B` that is dominated at least by one
solution in `A`.

If `C(A, B) = 1`, then every solution in `B` is dominated
by a solution in `A`, while `C(A, B) = 0` means that no solution in `B` is
dominated by a solution in `A`.

This metric, also known as C-metric, is a non-symmetric, cycle-inducing
metric. Compared to the S-metric has lower computational overhead, and scale
and reference point indepedent. If the approximation sets are not evenly
distributed, the results are unreliable [2].
"""
coverage(A::ParetoFront, B::ParetoFront) =
    sum([1 for j in nsols(A)
              if weaklyDominatesAny(solution(A, j), B)]) / nsols(B)

r1r(T::ParetoFront, A::ParetoFront) =
    error("Metric r1r is not implemented yet.")

r2r(T::ParetoFront, A::ParetoFront) =
    error("Metric r2r is not implemented yet.")


r3r(T::ParetoFront, A::ParetoFront) =
    error("Metric r3r is not implemented yet.")



# References

# [1] - Zitzler, E. (1999). Evolutionary Algorithms for Multiobjective
# Optimization: Methods and Applications. Swiss Federal Institute of Technology.

# [2] - Knowles, J., and Corne, D. (2002). On Metrics for Comparing Nondominated
# Sets. In Proceedings of the 2002 Congress on Evolutionary Computation,
# CEC 2002 (pp. 711–716).

# [3] - Veldhuizen, D. V. (1999). Multi Objective evolutionary algorithms:
# Classifications, Analysis, New Innovations. Multi Objective evolutionary
# algorithms. Air Force Institute of Technology, Wright Patterson, Ohio.

# [4] - Knowles, J. D. (2002). Local-Search and Hybrid Evolutionary Algorithms
# for Pareto Optimization, (February).

# [5] - Schott, J. R. (1995). Fault Tolerant Design Using Single and
# Multicriteria Genetic Algorithm Optimization. Massachusetts Institute of
# Technology, Boston, MA.

end
