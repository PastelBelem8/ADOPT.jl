using DelimitedFiles
using Statistics

using Main.MscThesis
using Main.MscThesis.Pareto

# -----------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------
readfile(filename, dlm=','; header=false) =
    open(filename, "r") do io
        if header readline(io); end
        readdlm(io, dlm, Float64, '\n')
    end
read_objectives(filename, cols=(:); dlm=',', header=false) =
    readfile(filename, dlm, header=header)[:,cols]'

writefile(X, filename, dlm=','; mode="w") =
    open(filename, mode) do io
        writedlm(io, X, dlm)
    end

result_files(algorithm, phase) =
    ["Results-Phase$phase/$algorithm/$(algorithm)_results0$i.csv" for i in 1:3]

# -----------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------
describe(results, quantiles=[0.25, 0.5, 0.75]) =
    let stats = Real[]
        push!(stats, minimum(results))
        push!(stats, maximum(results))
        push!(stats, Statistics.mean(results))
        push!(stats, Statistics.std(results))

        [push!(stats, q) for q in Statistics.quantile(results, quantiles)]
        stats
    end
describe_header(quantiles=[0.25, 0.5, 0.75]) =
    append!(["Min", "Max", "Mean", "Stdev"], map(q -> "Quantile$q", quantiles))

# -----------------------------------------------------------------------
# Pareto Front
# -----------------------------------------------------------------------
"Receives a list of matrices and returns the Non dominated solutions within the matrices"
create_pareto_front(Xs, ys) =
    let nobjs = size(ys[1], 1)
        nvars = 1
        vars(_, _) = nothing
        if Xs === nothing
            vars(_, y) = ones((1, size(y, 2)))
        else
            vars(X, _) = X
            nvars = size(Xs[1], 1)
        end
        pareto_front = Pareto.ParetoResult(nvars, nobjs)

        map((X, y) -> push!(pareto_front, X, y), Xs, ys)
        pareto_front
    end


"Receives a list of filenames and retrieves the pareto front "
collect_pareto_front(algorithms, objs) = let
        get_data(filename) = read_objectives(filename, objs, header=true)

        if isfile("outputs/pf/collect_pf_algorithms.txt")
            throw(ErrorException("You have already ran the method to collect the PF from all files, if you wish to proceed, please remove the files under the outputs/pf directory."))
        end

        filenames = filter(isfile, collect(Iterators.flatten([result_files(a, i) for (a,i) in Iterators.product(algorithms, 1:7)])))
        ys = map(get_data, filenames)
        pf = create_pareto_front(nothing, ys)

        ds = Pareto.dominated_objectives(pf)
        nds = Pareto.nondominated_objectives(pf)
        # Compute max_min
        ds_min = mapslices(minimum, ds, dims=2)[:]
        nds_min = mapslices(minimum, nds, dims=2)[:]

        ds_max = mapslices(maximum, ds, dims=2)[:]
        nds_max = mapslices(maximum, nds, dims=2)[:]

        writefile(algorithms, "outputs/pf/collect_pf_algorithms.txt")
        writefile(ds, "outputs/pf/pf_dominated-ALL.csv")
        writefile(nds, "outputs/pf/pf_nondominated-ALL.csv")
        writefile(hcat(ds_min, nds_min, ds_max, nds_max), "outputs/pf/min_max.txt")
        nds # ndims x nsamples
    end


get_pareto_front(filenames) = let
    get_vars(filename) = readfile(filename, header=true)[:, [4, 5, 6, 7, 8, 9]]'
    get_objs(filename) = readfile(filename, header=true)[:, [10, 11]]'

    Xs = map(get_vars, filenames)
    ys = map(get_objs, filenames)
    println(size(Xs), size(ys))
    create_pareto_front(Xs, ys)
end

# -----------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------
"Receives an indicator and an algorithm and computes the metric for the specified algorithm in a specific phase"
Base.run(algorithm, phase, indicator, objs) = let
    files = result_files(algorithm, phase)

    pfs = map(Pareto.nondominated_objectives,
                map(create_pareto_front,
                    [[read_objectives(file, objs, header=true)] for file in files]))
    results = map(indicator, pfs)
    results, describe(results)
    end

collect_stats_by_phase(algorithm, indicators_labels, indicators_fs, phase, objs) =
    if isempty(filter(isfile, result_files(algorithm, phase))) return;
    else
            quantiles = [0.25, 0.50, 0.75]
            stats = describe_header(quantiles)
            headers = vcat("Indicators", stats)
            headers = reshape(headers, (1, length(headers)))

            stats = Matrix(undef, length(indicators_labels), length(headers)-1)
            for (i, indicator) in enumerate(indicators_fs)
                _, stats[i,:] = run(algorithm, phase, indicator, objs)
            end
            writefile(vcat(headers, hcat(indicators_labels, stats)), "outputs/stats/Results_Stats_$(algorithm)_ph$(phase).csv")
            stats

    end

collect_stats(algorithms, indicators_labels, indicators_fs, phases, objs) =
    for algorithm in algorithms
        for phase in phases
            println("$algorithm, $phase")
            collect_stats_by_phase(algorithm, indicators_labels, indicators_fs, phase, objs)
        end
    end

group_results_by_phase(algorithm, phase, objs) =
    let files = filter(isfile, result_files(algorithm, phase))
        if !isempty(files)
            Xs = map(file -> read_objectives(file, objs, header=true), files)
            X = hcat(Xs...)
            writefile(X', "outputs/plots/$(algorithm)_ph$(phase)_joined.csv")
        end
    end
group_results(algorithms, phases, objs) =
    for algorithm in algorithms
        for phase in phases
            group_results_by_phase(algorithm, phase, objs)
        end
    end

# ######################## Global names ########################
objs_cols = [10, 11]
algorithms = ["IBEA", "EpsMOEA", "MOEAD", "NSGAII", "OMOPSO", "PAES", "PESA2", "SMPSO", "SPEA2"]
filename = "OMOPSO_results_01.csv"

pf = get_pareto_front([filename])

writedlm("PF_nondominated_vars.csv", Main.MscThesis.Pareto.nondominated_variables(pf), ',')
writedlm("PF_nondominated_objs.csv", Main.MscThesis.Pareto.nondominated_objectives(pf), ',')

#
# # True Pareto Front
# PF = try
#         collect_pareto_front(algorithms, objs_cols)
#     catch y
#         if isa(y, ErrorException)
#             readfile("outputs\\pf\\pf_nondominated-ALL.csv", header=false)
#         else
#             throw(y)
#         end
#     end
#
# pf_mins = read_objectives("outputs/pf/min_max.txt", [1, 2], header=false)'
# PF_min = mapslices(minimum, pf_mins, dims=2)[:]
# pf_maxs = read_objectives("outputs/pf/min_max.txt", [3, 4], header=false)'
# PF_max = mapslices(maximum, pf_maxs, dims=2)[:]
#
# hv(X) = Main.MscThesis.unitScale(X, PF_min, PF_max) |> hypervolumeIndicator
#
# # Run each indicator
# independent_indicators = [
#                             hv,
#                             onvg,
#                             spacing,
#                             spread,
#                             maximumSpread
#                         ]
#
# reference_indicators =  [
#                             onvgr,
#                             errorRatio,
#                             maxPFError,
#                             GD,
#                             IGD,
#                             d1r,
#                             M1
#                             ]
#
# reference_indicators = map(i -> (A -> i(PF, A)), reference_indicators)
# # comparative_indicators = [coverage, epsilonIndicator, additiveEpsilonIndicator, R1, R2, R3]
#
# indicators_labels = [
#                         "HV",
#                         "ONVG",
#                         "SPACING",
#                         "SPREAD",
#                         "MAXIMUMSPREAD",
#                         "ONVGR",
#                         "ERRORRATIO",
#                         "MAXPFError",
#                         "GD",
#                         "IGD",
#                         "D1R",
#                          "M1"
#                         ]

# indicators_functions = vcat(independent_indicators, reference_indicators)

# Create Files to plot PF
# group_results(algorithms, collect(1:7), objs_cols)
# collect_stats(algorithms, indicators_labels, indicators_functions, collect(1:7), objs_cols)

#= Test
readfile("Results-Phase1/IBEA/IBEA_results01.csv", header=true)
read_objectives("Results-Phase1/IBEA/IBEA_results01.csv", objs_cols, header=true)
result_files("IBEA", 1)

A = [[1 2 2; 3 1 4], [3 4; 6 5]]
p = load_pareto_front(A)
ND = Pareto.nondominated_objectives(p)
writefile(ND, "PARETO_FRONT.csv")

hv(PF)
run(algorithms[1], 5, independent_indicators[1], objs_cols)
collect_stats_by_phase(algorithms[1], indicators_labels, indicators_functions, 1, objs_cols)
collect_stats_by_phase("NSGAII", indicators_labels, indicators_functions, 4, objs_cols)
group_results_by_phase("IBEA", 1, objs_cols)


# Collect Parameters and Objectives
# pf = get_pareto_front(algorithms)
# writedlm("PF_nondominated_vars.txt", Main.MscThesis.Pareto.nondominated_variables(pf), ',')
# writedlm("PF_nondominated_objs.txt", Main.MscThesis.Pareto.nondominated_objectives(pf), ',')
=#
