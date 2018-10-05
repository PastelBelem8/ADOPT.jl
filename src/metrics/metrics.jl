module Metrics

"Returns the `distance` between vector `p` and the nearest point in `M`."
min_distance(M, p, distance=Distances.euclidean) =
    minimum(mapslices(m -> distance(m, p), M, dims=2), dims=1)[1]


generational_distance(T, A, distance) =
    let
        n = size(A)[1]
        squared_dists = mapslices(row -> min_distance(T, row, distance)^2, A; dims=2)
        sqrt(sum(squared_dists)) / n
    end
end

inverse_generational_distance(T) = print(T)

error_ratio(T, A) = 1 - sum(mapslices(a -> numberIntersections(a, T), A, dims=2)) / size(A)[1]

# FIXME - Especializar p/ Arrays/Vectors vs Matrizes..
intersections(a, A) = mapslices(v -> v == a, A, dims=2)
numberIntersections(a, A) = count(numberIntersections())





# Test
# Base.in(x) = Base.Fix2(in, x)
# Base.in(a::Matrix{Number}, A::Matrix{Number,2}) = any(iszero.(sum(a .- A, dims=2)))
#
# @which [1 2] in [1 2; 3 5]
# [1 2] in [1 2; 3 4]



M = [0 2; 0.5 1.5; 1.75 1; 2 0.5]
A1 = [2 1; 0.5 3]
A2 = [0.6 1.5;]
A3 = [0.7 1.5; 0.1 2.1; 1.8 1]

min_distance(M, A1[1, :], Distances.euclidean)
min_distance(M, A1[2, :], Distances.euclidean)
min_distance(M, A2[1, :], Distances.euclidean)

# mapslices(a -> min_distance(M, a, Distances.euclidean), A1; dims=2)
generational_distance(M, A1, Distances.euclidean)
generational_distance(M, A2, Distances.euclidean)
generational_distance(M, A3, Distances.euclidean)

M2 = [3 4; 0.5 2; 3 4; 5 -2]
A = [4 4; 3 4; 5 -2]
error_ratio(M2, A)
end
