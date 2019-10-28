"""This function applies different indicators per
iteration, returning the result of applying each
indicator to a subset of the data of size
``iteration_size``.

Data is in the m x n format, where m is the number of records and
n the number of objectives.

Examples:
julia> indicator = x -> sum(x)
#9 (generic function with 1 method)
julia> compute_per_iteration(a, Dict(:sum => indicator), 1)
Dict{Any,Any} with 1 entry:
  :sum => Real[3, 7]
"""
compute_per_iteration(data, iter_size, indicators, senses, mins, maxs) =
let normalized_data = normalize_data(data, senses, mins, maxs),
    new_senses = [:MAX for _ in senses],
    n_dims = size(normalized_data, 1),
    n_evals = size(data, 2),
    nd_results = Dict(name => Float64[] for (name, _) in indicators),
    pf = Pareto.ParetoResult(1, n_dims, new_senses)

    for i in 1:iter_size:n_evals
        limit = min(i + iter_size - 1, n_evals)
        data = normalized_data[:, i:limit]
        push!(pf, ones(1, size(data, 2)), data)

        nd_sols = Pareto.nondominated_objectives(pf)
        for (indicator_name, indicator_func) in indicators
            indicator_value = indicator_func(nd_sols)
            append!(nd_results[indicator_name], indicator_value)
        end
    end
    nd_results
end
# Matrix should be mxn, where m is the number of objectives and n the number
# of samples
normalize_data(A, senses, mins, maxs) =
let n_dims = size(A, 1),
    mins = copy(mins),
    maxs = copy(maxs),
    A_matrix = copy(A)
    # Step 1. Transform to maximization
    for (i, sense) in enumerate(senses)
        if sense in (:MIN, :MINIMIZE)
            A_matrix[i,:] = A[i,:] .* -1
            mins[i], maxs[i] = maxs[i] .* -1, mins[i] .* -1
        end
    end

    # Step 2. Scale (to ensure they lie in the positive)
    unit_scale = (a, min, max) -> (a .- min) ./ (max - min)
    for i in 1:n_dims
        A_matrix[i,:] = unit_scale(A_matrix[i,:], mins[i], maxs[i])
    end
    if size(A) != size(A_matrix)
        throw(DimensionMismatch("Expected: $(size(A)) but obtained: $(size(A_matrix))"))
    elseif any(A_matrix .> 1)
        throw(DomainError("Found values above 1 after scaling: $A_matrix"))
    end

    A_matrix
end





feasible_ratio(v::Vector) = length(filter(x -> x == "FALSE", v)) / length(v)
unfeasible_ratio(v::Vector) = 1 - feasible_ratio(v)

export compute_per_iteration
