using DelimitedFiles
using Statistics

using Main.MscThesis.Pareto
using Main.MscThesis.Indicators

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
create_pareto_front(Xs) =
    let nobjs = size(Xs[1], 1)
        vars(n) = ones((1, n))
        pf = Pareto.ParetoResult(1, nobjs)

        for X in Xs
            push!(pf, vars(size(X, 2)), X)
        end

        pf
    end

"Receives a list of filenames and retrieves the pareto front "
collect_pareto_front(algorithms, objs) = let
        get_data(filename) = read_objectives(filename, objs, header=true)

        if isfile("outputs/pf/collect_pf_algorithms.txt")
            throw(ErrorException("You have already ran the method to collect the PF from all files, if you wish to proceed, please remove the files under the outputs/pf directory."))
        end

        filenames = filter(isfile, collect(Iterators.flatten([result_files(a, i) for (a,i) in Iterators.product(algorithms, 1:7)])))
        Xs = map(get_data, filenames)
        pf = create_pareto_front(Xs)

        writefile(algorithms, "outputs/pf/collect_pf_algorithms.txt")
        writefile(Pareto.dominated_objectives(pf), "outputs/pf/pf_dominated-ALL.csv")
        writefile(Pareto.nondominated_objectives(pf), "outputs/pf/pf_nondominated-ALL.csv")
        Pareto.nondominated_objectives(pf) # ndims x nsamples
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
        let stats = describe_header(quantiles)
            headers = vcat("Indicators", stats)
            headers = reshape(headers, (1, length(headers)))

            stats = Matrix(undef, length(indicators_labels), length(headers)-1)
            for (i, indicator) in enumerate(indicators_fs)
                _, stats[i,:] = run(algorithm, phase, indicator, objs)
            end
            writefile(vcat(headers, hcat(indicators_labels, stats)), "outputs/stats/Results_Stats_$algorithm.csv")
            stats
        end
    end

collect_stats(algorithms, indicators_labels, indicators_fs, phases, objs) =
    for algorithm in algorithms
        for phase in phases
            collect_stats_by_phase(algorithm, indicators_labels, indiators_fs, phases, objs)
        end
    end

group_results_by_phase(algorithm, phase, objs) =
    let files = filter(isfile, result_files(algorithm, phase))
        if !isempty(files)
            Xs = map(file -> read_objectives(file, objs, header=true), files)
            X = hcat(Xs...)
            writefile(X', "outputs/$(algorithm)_ph$(phase)_joined.csv")
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

# True Pareto Front
PF = try
        collect_pareto_front(algorithms, objs_cols)
    catch
        readfile("outputs\\pf\\pf_nondominated-ALL.csv", header=false)
    end

PF_min = mapslices(minimum, PF, dims=2)[:]
PF_max = mapslices(maximum, PF, dims=2)[:]

X = Main.MscThesis.unitScale(PF, PF_min, PF_max)
Main.MscThesis.Indicators.hypervolumeIndicator(X)

# Run each indicator
independent_indicators = [
                            # hypervolumeIndicator,
                            Indicators.onvg,
                            Indicators.spacing,
                            Indicators.spread,
                            Indicators.maximumSpread
                        ]
#
# reference_indicators =  [
#                             Indicators.onvgr,
#                             Indicators.errorRatio,
#                             Indicators.maxPFError,
#                             Indicators.GD,
#                             Indicators.IGD,
#                             Indicators.d1r,
#                             Indicators.M1
#                             ]
#
# reference_indicators = map(i -> (A -> i(PF, A)), reference_indicators)
# # comparative_indicators = [coverage, epsilonIndicator, additiveEpsilonIndicator, R1, R2, R3]
#
# indicators_labels = [
#                         # "HV",
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
#
# indicators_functions = vcat(independent_indicators, reference_indicators)
#
# # Create Files to plot PF
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

collect_stats_by_phase(algorithms[1], indicators_labels, indicators_functions, 1, objs_cols)
group_results_by_phase("IBEA", 1, objs_cols)

=#
