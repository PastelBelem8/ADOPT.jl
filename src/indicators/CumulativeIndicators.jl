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
compute_per_iteration(data, indicators, iteration_size) =
let data = data'
    n_evals = size(data, 2)
    n_iterations = div(n_evals, iteration_size)

    results = Dict()
    for (indicator_name, indicator_func) in indicators
        results[indicator_name] = Real[]
        for i in 1:n_iterations
            results[indicator_name] = [
                results[indicator_name];
                indicator_func(data[:, i:i+iteration_size-1])
            ]
        end
    end
    results
end


feasible_ratio(a::Vector) = length(filter(v -> v == "FALSE", a))
unfeasible_ratio(a::Vector) = 1 - feasible_ratio(a)

export compute_per_iteration
