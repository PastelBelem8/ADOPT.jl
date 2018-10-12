module MOOIndicators

# Define the type ParetoFront

mutable struct ParetoFront
    values::AbstractMatrix
end


# Private ----------------------------------------------------------------
ParetoFront(dim::Int64) =
    dim > 0 ? ParetoFront(Array{Float64}(undef, dim, 0))
            : thow(ArgumentError("Invalid value for dim argument: $dim.
                Consider specifying a value ≥0."))

"Modifies a Pareto Optimal solution `v` to the ParetoFront `PF`."
function addpareto!(v::AbstractVector, PF::ParetoFront)
    assertDimensions(v, PF)
    deletedominated!(v, PF)
    PF.values = [v PF.values]
end

assertDimensions(A::ParetoFront, B::ParetoFront) =
    if nrows(A) != nrows(B)
        throw(DimensionMismatch("Different objective-space dimensions:
                                    $(nrows(A)) != $(nrows(B))."))
    end
assertDimensions(a::AbstractVector, A::ParetoFront) =
    if length(a) != nrows(A)
        throw(DimensionMismatch("Different objective-space dimensions:
                                    $(length(a)) != $(nrows(A))."))
    end
assertDimensions(a::AbstractVector, b::AbstractVector) =
    if length(a) != length(b)
        throw(DimensionMismatch("Different objective-space dimensions:
                                    $(length(a)) != $(length(b))."))
    end

contains(P::ParetoFront, v::AbstractVector) = contains(P.values, v)
contains(V::AbstractMatrix, v::AbstractVector) =
    any([v == V[:,j] for j in 1:ncols(V)])

empty(P::ParetoFront) = isempty(P.values)

function order(V::AbstractMatrix)
    indices = Vector(1:ncols(V))
    lt_f = (i0, i1) -> V[:, i0] < V[:, i1]
    sort!(indices,lt=lt_f)
    V[:,indices]
end

@inline ndim(P::ParetoFront) = nrows(P.values)
@inline nsols(P::ParetoFront) = ncols(P.values)
@inline solution_obj(P::ParetoFront, i::Int, j::Int) = P.values[i,j]
@inline solution(P::ParetoFront, j::Int) = P.values[:,j]
@inline solutions(P::ParetoFront) = P.values
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

"Returns whether there is a `v1` belonging to `V`, that is weakly dominated by
`v`."
weaklyDominatesAny(v::AbstractVector, V::AbstractArray) =
    any([weaklyDominates(v, V[:, j]) for j in 1:ncols(V)])

"Returns whether `v0` is never worse than `v1` and has an objective value that
is at least better than `v1`."
weaklyDominates(v0::AbstractVector, v1::AbstractVector) = v0 ≤ v1 && v0 < v1


# Public -----------------------------------------------------------------
"""
    add!(v, PF)

Adds the `v` to the set of optimal vectors `PF` if `v` is Pareto Efficient.
Before adding `v` to `PF`, it is necessary to remove dominated solutions.
If `v` is not Pareto Efficient, then `PF` will not be changed.
"""
add!(v::Vector, PF::ParetoFront) = isparetoOptimal(v, PF) ?
                                        addpareto!(v, PF) : PF

"Deletes the vectors in `PF` dominated by `v`."
deletedominated!(v::Vector, PF::ParetoFront) =
    begin
        nd_ix = [j for j in 1:j if !weaklyDominates(v, solution(PF, j))]
        PF.values = solutions([:, nd_ix])
    end
Base.show(io::IO, pf::ParetoFront) =
    print(io, [solution(pf, j) for j in 1:nsols(pf)]

# ------------------------------------------------------------------------
# Multi-Objective Optimization Indicators
# ------------------------------------------------------------------------

# Depedencies ------------------------------------------------------------
using Distances
using Statistics
using LinearAlgebra

# ------------------------------------------------------------------------
# Independent Indicators
# ------------------------------------------------------------------------
"""
    hypervolumeIndicator(r, A) -> s

Returns the hypervolume of the multi-dimensional region enclosed by `A` and a
reference point `r`, i.e., computes the size of the region dominated by `A`
[1]. It is an independent Indicator, that induces a complete ordering, and it
is non-cardinal. Different values of `r` might influence differently the
scores [2, 5]. By default the reference point will be determined based on
the worst known value in each objective and shift it by a small amount. A
set with larger hypervolume is likely to present a better set of trade-offs
than sets with lower hypervolume.

Moreover, also known as S-metric or Lebesgue measure, the hypervolume
indicator (HV or HVI) is classified as NP-hard in the number of objectives,
evidencing the complexity of the Indicator (unless P=NP, no polynomial
algorithm for HV exists).

We provide a current state-of-the art implementation based on Badstreet's
Phd Thesis [6]. Motivated by the bad performance of currently available
algorithms, as well as the high complexity of HV, Badstreet proposed two
algorithms aimed at improving the feasibility of HV in multi-objective
optimisation:
    (1) IHSO (Incremental Hypervolume by Slicing Objectives)
    (2) IIHSO (Iterated IHSO)
Due to the high computational impact, this Indicator is often infeasible for
problems with many objectives or for problems with large data sets.
"""
function hypervolumeIndicator(r::AbstractVector, A::ParetoFront)
    assertDimensions(r, A)
    hvol = 0
    for j in 1:nsols(A)
        # Compute Hypervolume
        hvol += abs(solution_obj(A, 1, j) - r[1]) *
                abs(solution_obj(A, 2, j) - r[2])
        # Update reference point to prevent overlapping volumes.
        r[2] = solution_obj(A,2, j) # FIXME - It won't probably work for 3D+ (must generalize)
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
approximation sets using exclusively this Indicator (e.g. approximation sets
with many points will always be better than sets with fewer points that
dominate all the points of the first Indicator). For more information, refer
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

Given an approximation set, computes the variance of the distances between
each point and its closest neighbor. The Set Spacing (SS) [5] is a measure
of the uniformity of the distribution of solutions in the approximation set.
A value of zero would mean that all members of e are equally spaced from
one another.

By default, it uses the the cityblock distance, also called
[Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) as
defined by Schott [5]. However, it is also possible to use the Euclidean
distance as proposed by Deb. et al [4] by calling [`Δ`](@ref) instead.

While inducing complete ordering and being based on the cardinality of `A`,
this Indicator has low computational overhead. The original proposed
non-normalized version of the Indicator [5] might be problematic as the
distances are not properly normalized. Consider the normalized version
spacing as proposed in [4] by specifying by using [`Δ`](@ref) or
[`debSpacing`](@ref).
"""
function spacing(A::ParetoFront)
    min_ds = [ minimum_distance(solution(A, j),
                                solution(A, 1:end .!=j),
                                Distances.cityblock)
                    for j in 1:nsols(A)]
    Statistics.var(min_ds)
end

"""
    Δ(A) -> s
    Spread(A) -> s

Computes the Deb's spacing Indicator using the Euclidean distance, which
measures the consecutive distances among the solutions in `A`.
"""
function Δ(A::ParetoFront)
    nsols = nsols(A)
    min_ds = [minimum_distance(solution(A, j), solution(A, 1:end .!=j))
                    for j in 1:nsols]
    mean_d = Statistics.mean(min_ds)
    sum(abs.(mean_d .- min_ds)) / (nsols-1)
end
spread(A::ParetoFront) = Δ(A)

"""
    m3(A) -> s
    maximumSpread(A) -> s
    overallParetoSpread(A) -> s

Computes the euclidean distance between the bounds of each objective dimension.

The Maximum Spread Indicator measures the extent of the search space in each
dimension, by calculating the Euclidean distance between the maximum and
minimum of each objective. A greater value of Maximum Spread indicates a
larger coverage of the search space [10].

See also: [`m1`](@ref)
"""
maximumSpread(A::ParetoFront) =
    let min_max_val = mapslices((a) -> [minimum(a), maximum(a)],
                                 solutions(A),
                                 dims=2)
        Distances.euclidean(min_max_val[:,1],min_max_val[:,2])
    end
m3(A::ParetoFront) = maximumSpread(A)
overallParetoSpread(A::ParetoFront) = maximumSpread(A)


"""
DM [15] are two very related diversity indicators that apply an entropy concept
to calculate the diversity of solutions. DM is based on the entropy indicator.
Both of them basically attempt to project the solutions of an approximation
set on a suitable hyperplane assigning them entropy functions that later will
be added together to compose a normalized entropy function.
"""
entropy(A::ParetoFront) =
    error("Entropy Indicator is not implemented yet.")

diversityMeasure(A::ParetoFront) =
    error("Indicator Dversity Measure is not implemented yet.")



# ------------------------------------------------------------------------
# Reference Indicators
# Reference Indicators require the existence of a True Pareto Front set in order
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
vector in `A` and the corresponding closest vector in `T`. This Indicator is
called Maximum Pareto Front Error (MPFE) and it was first introduced by
Veldhuizen [3] in 1999. It is a reference Indicator that induces ordering and
it is a non-cardinal Indicator.

The lowest the value of `e` the better the approximation set.
"""
maxPFError(T::ParetoFront, A::ParetoFront) =
    let min_ds = [minimum_distance(solution(A, j), T)
                    for j in 1:nsols(A)]
        maximum(min_ds)
    end

minimum_distance(v::AbstractVector, V::AbstractMatrix,
                    metric::Function=Distances.euclidean) =
    minimum(distances(v, V, metric))
@inline distances(v::AbstractVector, V::AbstractMatrix, metric::Function) =
    [metric(solution(V,j), v) for j in 1:ncols(V)]

"""
    GD(T, A) -> s
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
#        associated to the GD Indicator that should be accounted for. [2], [3]
generationalDistance(T::ParetoFront, A::ParetoFront) =
    let nsols = nsols(A)
        squared_min_dists = [minimum_distance(solution(A, j), T)^2
                                for j in 1:nsols]
        √(sum(squared_min_dists)) / nsols
    end

GD(T, A) = generationalDistance(T,A)

"""
    IGD(T, A) -> s
    invertedGD(T, A) -> s

Returns the average distance from each reference point in `T` to the nearest
solution in `A`.

When a well distributed true Pareto Front `T` is given, smaller values for this
Indicator suggest the good convergence of solutions of `A`.

See also: [`generationalDistance`](@ref)
"""
invertedGenerationalDistance(T::ParetoFront, A::ParetoFront) =
    generationalDistance(A, T)

IGD(T, A) = invertedGenerationalDistance(T, A)


"""
    d1r(T, A) -> s

Measures the mean distance, over the points in the reference set `T`, of
the nearest point in an approximation set `A`.

Can be used to measure diversity and convergence.
"""
d1r(T::ParetoFront, A::ParetoFront) =
    let nsols = nsols(T)
        Λ = objs_range(T)
        d = (a, t) -> maximum((t - a) * Λ)
        min_dists = [minimum_distance(solution(T, j), solutions(A), d)
                        for j in 1:nsols(T)]
        sum(min_dists) / nsols
    end

objs_range(P::ParetoFront) =
    let solutions = solutions(P)
        ranges = abs(maximum(solutions, dims = 2) - minimum(solutions, dims=2))
        1 ./ ranges
    end

"""
    M1(T, A) -> s
    averageDistance(T, A) -> s

Returns the average distance of the approximation set `A` to the reference
set `T`.

M1*, here denoted as M1, is a reference Indicator that computes an averaged
approximation of the solutions in the approximation set to the nearest points
in a reference set `T` (ideally representing the true Pareto Front). For
further information refer to [11].
"""
M1(T::ParetoFront, A::ParetoFront) =
    let nsols = nsols(A)
        sum([minimum_distance(solution(A, j), T) for j in 1:nsols]) / nsols
    end

averageDistance(T::ParetoFront, A::ParetoFront) = M1(T, A)

# ------------------------------------------------------------------------
# Direct Comparative Indicators
# ------------------------------------------------------------------------
"""
    coverage(A, B) -> s

Returns the fraction of solutions in `B` that is dominated at least by one
solution in `A`.

If `C(A, B) = 1`, then every solution in `B` is dominated
by a solution in `A`, while `C(A, B) = 0` means that no solution in `B` is
dominated by a solution in `A`.

This Indicator, also known as C-metric, is a non-symmetric, cycle-inducing
Indicator. Compared to the S-metric has lower computational overhead, and scale
and reference point indepedent. If the approximation sets are not evenly
distributed, the results are unreliable [2].
"""
coverage(A::ParetoFront, B::ParetoFront) =
    sum([1 for j in nsols(A)
              if weaklyDominatesAny(solution(A, j), B)]) / nsols(B)

"""
    epsilonIndicator(A, B) -> s

Given two approximation sets `A` and `B`, returns the smallest amount, ϵ, that
must be used to translate the set `A` so that every point in `B` is covered [8].

In other words, the epsilon indicator gives the factor by which an approximation
set is worse than another with respect to all objectives [9].

See also: [`additiveEpsilonIndicator`](@ref).
"""
function epsilonIndicator(T::ParetoFront, A::ParetoFront)
    epsilon_aux(T, A, (x, y) -> x ./ y)
end

"""
    additiveEpsilonIndicator(T, A) -> ϵ

Given a reference set `T` and an approximation set `A`, returns a value ϵ
representing the largest error in the objective space between a solution in `A`
and a solution in `B`. In other words, it computes the largest difference by
which an approximation set is worse than another with respect to all objectives.

A small value on this Indicator indicates that the solutions in `A` are close to
the reference set `T`, which ideally would be the True Pareto Front.

See also: [`epsilonIndicator`](@ref)
"""
additiveEpsilonIndicator(T::ParetoFront, A::ParetoFront) =
    epsilon_aux(T, A, (x, y) -> abs.(x .- y))
end

# Works as the pattern template method
"Computes the epsilon indicator of two sets using the `eps_f`. It is an
auxiliar function to be used by the real epsilon indicators (e.g, additive,
multiplicative)."
function epsilon_aux(T::ParetoFront, A::ParetoFront, eps_f::Function)
    assertDimensions(T, A)
    nA, nT, ε = nsols(A), nsols(T), zeros((nA, nT))

    for (i, j) in Iterators.product(1:nA, 1:nT)
        ε[i,j] = maximum(eps_f(solution(A, i), solution(T, j)))
    end

    maximum([pairwiseDifferences(ε[:,j], minimum) for j in 1:nT])
end

"Computes the pairwise differences between elements of `A` and selects specific
pairwise differences according to f."
pairwiseDifferences(A::AbstractVector, f::Function) =
    let length = length(A) - 1
        results = Vector(length)
        for i in 1:length
            results[i] = f(abs(A[i] .- A[(i+1):end])
        end
        f(results)
    end


"""
    R1(A, T, U, P) -> s
    R2(A, T, U, P) -> s
    R3(A, T, U, P) -> s

Given a set `U` of utility functions and its probability vector `P`,
return the set that will most likely provide higher utility.

The binary indicators `R1`, `R2` and `R3` belong to the family of R-metrics
[2, 7].
[`R1`](@ref) calculates the probability that `A` is better than `T` over a set
of utility functions.
[`R2`](@ref) calculates the expected difference in the utility of an
approximation `A` with another one `T`.
[`R3`](@ref) calculates the ratio of the differences in the utility of an
approximation `A` with another one `T`.

All the R-metrics are non-cardinal Indicators that might be direct comparative or
reference Indicators, depending whether `T` represents the true Pareto Front or
just an approximation. When passing the true Pareto Front, R-metrics induce
order. Additionally, these Indicators are scaling independent and have lower
computational overhead than the [`hypervolumeIndicator`](@ref).
"""
# Refine description of R Indicators.
function Rmetric(A::ParetoFront, B::ParetoFront,
                    U::Vector{Function}, P::Vector{Float64}, λ::Function)
    assertDimensions(U, P)
    sum([λ(A, T, u[i]) * pu[i] for i in length(u)])
end

"Retrieves ratio of one set `A` having better utility than the set `T`."
function R1(T::ParetoFront, A::ParetoFront, u::Vector{Function}, P::Vector{Float64})
    function λ(V0::ParetoFront, V1::ParetoFront, f::Function)
        v0_max_u = maximum([u(solution(V0, j)) for j in 1:nsols(V0)])
        v1_max_u = maximum([u(solution(V1, j)) for j in 1:nsols(V1)])

        if v0_max_u > v1_max_u
            1
        elseif v0_max_u == v1_max_u
            0.5
        else
            0
        end
    end

    Rmetric(A, T, U, P, λ)
end

"Retrieves the expected mean difference in utilities of set `A` and `T`."
function R2(T::ParetoFront, A::ParetoFront, u::Vector{Function}, P::Vector{Float64})
    function λ(V0::ParetoFront, V1::ParetoFront, f::Function)
        v0_max_u = maximum([u(solution(V0, j)) for j in 1:nsols(V0)])
        v1_max_u = maximum([u(solution(V1, j)) for j in 1:nsols(V1)])

        v0_max_u - v1_max_u
    end

    Rmetric(A, T, U, P, λ)
end

"Retrieves the expected mean relative difference in utilities of set `A` and `T`."
function R3(T::ParetoFront, A::ParetoFront, u::Vector{Function}, P::Vector{Float64})
    function λ(V0::ParetoFront, V1::ParetoFront, f::Function)
        v0_max_u = maximum([u(solution(V0, j)) for j in 1:nsols(V0)])
        v1_max_u = maximum([u(solution(V1, j)) for j in 1:nsols(V1)])

        1 - v0_max_u/v1_max_u
    end

    Rmetric(A, T, U, P, λ)
end


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

# [7] - Hansen, M. P., and Jaszkiewicz, A. (1998). Evaluating the quality of
# approximations to the non-dominated set. IMM Technical Report IMM-REP-1998-7.


# [8] - Coello, C. a C., Lamont, G. B., and Veldhuizen, D. a Van. (2007).
# Evolutionary Algorithms for Solving Multi-Objective Problems Second Edition.
# Design.

# [9] - Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., and da Fonseca,
# V. G. (2003). Performance assesment of multiobjective optimizers: an analysis
# and review. Evolutionary Computation, 7(2), 117–132.

# [10] - Bhuvaneswari M.C., Subashini G. (2015) Scheduling in Heterogeneous
# Distributed Systems. In: Bhuvaneswari M. (eds) Application of Evolutionary
# Algorithms for Multi-objective Optimization in VLSI and Embedded Systems.
# Springer, New Delhi

# [11] - Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of
# multiobjective evolutionary algorithms: empirical results. Evolutionary
# Computation, 8(2), 173–195.

end
