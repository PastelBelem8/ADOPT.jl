using DelimitedFiles
using Statistics

using Main.MscThesis
using Main.MscThesis.Pareto
# using Main.MscThesis.Indicators

base_folder = "C:\\Users\\catar\\Dropbox\\Work\\Thesis\\case-studies\\Robot\\final"
analysis_folder = "$(base_folder)/Analises"

outputs_folder = "$(base_folder)/outputs"
algorithms_folder = "$(outputs_folder)/algorithms"


pf_file = "$(outputs_folder)/non_dominated_o1_asc.csv"
pf_header = false

n_runs = collect(1:3)
algorithms = ["SPEA2", "NSGAII", "OMOPSO", "SMPSO", "EpsMOEA", "PAES", "PESA2", "MOEAD"]
n_algorithms = length(algorithms)
algorithms_header = true

# Variables Columns
vars_cols = collect(4:9)

# Objectives Columns
objs_cols = collect(10:11)
pf_objs_cols = collect(7:8)

read_cols(filename, dlm=','; objs=(:), header=false) =
    open(filename, "r") do io
        if header readline(io); end
        readdlm(io, dlm, Float64, '\n')[:, objs]
    end

# Collect Pareto Front
println("Collecting Pareto Front\nReading $(pf_objs_cols) from $(pf_file)...")
combined_pareto_front = read_cols(pf_file, objs=pf_objs_cols, header=pf_header)
combined_pareto_front = combined_pareto_front'
println("The combined pareto front is composed of $(size(combined_pareto_front, 1)) non-dominated solutions.")

front_min = [0.576628802, 0]
front_max = [1.802384017, 88.89642472]

# Approximated values
# front_min = [0.55, 0]
# front_max = [1.81, 89]

A = [1.3078 72.4613;
     0.9691 55.4516]'
unitScale(A, front_min, front_max)

# Get files per algorithm
result_files(algorithm) = ["$(analysis_folder)/$(algorithm)_results_0$i.csv" for i in n_runs]
result_files(algorithms[1]) # Test

# Get all files per algorithm
algorithms_files = map(result_files, algorithms)

# Expects nsamples x ndims format
create_pareto_front(y) =
    if isempty(y) return; else
        let pf = Pareto.ParetoResult(1, size(y, 2))
            push!(pf, ones(1, size(y, 1)), y')
            pf
        end
    end

# For each file create a pareto front
create_pareto_result(filename::String; objs=objs_cols, header=algorithms_header) =
    let y = read_cols(filename, objs=objs, header=header)
        create_pareto_front(y)
    end

create_pareto_result(algorithms_files[1][1]) # Test

# Collect for each file the results
algorithm_pareto_results = map(i -> map(Pareto.nondominated_objectives ∘ create_pareto_result, algorithms_files[i]),
                                1:length(algorithms))

# Indicators
# Independent Indicators
hv(X) = let y = Main.MscThesis.unitScale(X, front_min, front_max)
            a = filter(b -> b > 1, y)
            hypervolumeIndicator(y)
        end

independent_indicators = [
                            hv,
                            onvg,
                            spacing,
                            spread,
                            maximumSpread
                        ]
# Reference Indicators
reference_indicators =  [
                            onvgr,
                            errorRatio,
                            maxPFError,
                            GD,
                            IGD,
                            d1r,
                            M1
                        ]
reference_indicators = map(indicator -> (A -> indicator(combined_pareto_front, A)), reference_indicators)
indicators = vcat(independent_indicators, reference_indicators)
n_indicators = length(indicators)

header =  join(["HV","ONVG", "SPACING", "SPREAD", "MAXIMUM SPREAD", "ONVGR", "Error Ratio", "Max Pareto Front Error", "GD", "IGD", "D1R", "M1", "Algorithm", "Run"], ',')
header
# Apply indicator to all algorithms
write("$(outputs_folder)/indicators.csv", header)

for a_i in 1:n_algorithms
    for r_i in n_runs
        println("Analysing run $(r_i) from algorithm $(algorithms[a_i])")
        results = [ind(algorithm_pareto_results[a_i][r_i]) for ind in indicators]
        results = vcat(results, algorithms[a_i], r_i)
        open("$(outputs_folder)/indicators.csv", "a") do io
            write(io, "\n")
            join(io, results, ',')
        end
    end
end



algorithms_combined_files = map(a -> "$(algorithms_folder)/all_$(a)_solutions.csv", algorithms)
algorithm_pareto_results_combined_runs = map(Pareto.nondominated_objectives ∘ create_pareto_result, algorithms_combined_files)
algorithm_pareto_results_combined_runs
# Apply indicator to all algorithms
# write("$(outputs_folder)/indicators_3_runs_combined.csv", header)
#
# for a_i in 1:n_algorithms
#         println("Analysing all runs combined for the algorithm $(algorithms[a_i])")
#         results = [ind(algorithm_pareto_results_combined_runs[a_i]) for ind in indicators]
#         results = vcat(results, algorithms[a_i])
#         open("$(outputs_folder)/indicators_3_runs_combined.csv", "a") do io
#             write(io, "\n")
#             join(io, results, ',')
#         end
# end
