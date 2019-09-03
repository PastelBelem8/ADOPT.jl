using Main.MscThesis
using Main.MscThesis.Pareto

using Dates
using DelimitedFiles
using Statistics

# --------------------------------- -----------------------------------
# Global constants
# --------------------------------- -----------------------------------
# Problem Modeling - definitions
vars_cols = collect(1:6) # Variables
objs_cols = collect(7:8) # Objectives

has_header = false

# Algorithms
nruns = collect(1:3)
algorithms = [  "OMOPSO", "SMPSO", "MOEAD", "SPEA2",
                "EpsMOEA", "PESA2", "PAES", "NSGAII",
                "GDE3", "CMAES", "GPR_NSGAII", "GPR_Random",
                "GPR_SMPSO", "MLP_NSGAII", "MLP_Random", "MLP_SMPSO",
                "RF_NSGAII", "RF_Random", "RF_SMPSO"]
nalgorithms = length(algorithms)

# File constants
base_folder = "C:\\Users\\catar\\Dropbox\\Work\\Thesis\\case-studies\\Robot\\final"
analysis_folder = "$(base_folder)/Analises"

output_folder = "$(base_folder)/outputs"
ndfiles_folder = "$(output_folder)/nondominated"

tpf_file = "$(ndfiles_folder)/nondominated_o1_asc.csv"
tpf_file_1 = "$(ndfiles_folder)/nondominated_r1_o1_asc.csv"
tpf_file_2 = "$(ndfiles_folder)/nondominated_r2_o1_asc.csv"
tpf_file_3 = "$(ndfiles_folder)/nondominated_r3_o1_asc.csv"

# --------------------------------- -----------------------------------
# Auxiliar function definitions
# --------------------------------- -----------------------------------
read_cols(filename, dlm=','; objs=(:), header=false) = open(filename, "r") do io
    if header readline(io); end
    readdlm(io, dlm, Float64, '\n')[:, objs]
end

read_pf(file, objs=objs_cols, has_header=has_header) = begin
    println("Collecting Pareto Front (objs: $(objs)) from $(file)...")
    pf = read_cols(file, objs=objs, header=has_header)
    println("The combined pareto front is composed of $(size(pf, 1)) nondominated solutions.")
    pf = pf'
    pf
end

create_pareto_result(filename::String; objs=objs_cols, header=has_header) =
    let create_pareto_front(y) = if isempty(y) return; else
            let pf = Pareto.ParetoResult(1, size(y, 2))
                push!(pf, ones(1, size(y, 1)), y')
                pf
            end
        end
        cols = read_cols(filename, objs=objs, header=header)
        create_pareto_front(cols)
    end

# Get files per algorithm
result_files(algorithm) = ["$(ndfiles_folder)/$(algorithm)_$(i)_o1_asc.csv" for i in nruns]
result_files(algorithms[1]) # Sanity check

# --------------------------------- -----------------------------------
# Main Program
# --------------------------------- -----------------------------------
# Global Pareto Front
tpf_total = read_pf(tpf_file)
tpf1 = read_pf(tpf_file_1)
tpf2 = read_pf(tpf_file_2)
tpf3 = read_pf(tpf_file_3)

tpfs =[tpf1, tpf2, tpf3]
# We need to consider all the evaluated solutions (because one can have solutions with larger values than the ones discovered by the pareto front )
tpf_min = [0.57, 0] # mapslices(minimum, tpf, dims=2)
tpf_max = [1.86, 89.0] # mapslices(maximum, tpf, dims=2)

algorithms_files = map(result_files, algorithms)

#= Sanity Checks
read_pf(algorithms_files[1][1])
create_pareto_result(algorithms_files[1][1])
=#

algorithm_pareto_results = map(i -> map(Pareto.nondominated_objectives ∘ create_pareto_result,
                                            algorithms_files[i]),
                                1:length(algorithms))

# --------------------------------- -----------------------------------
# Indicators
# --------------------------------- -----------------------------------

unitScale(a::AbstractVector, min::Number, max::Number) = (a .- min) ./ (max - min)
unitScale(A::AbstractMatrix, min::AbstractVector, max::AbstractVector) = begin
    if size(A, 1) != length(min) || length(min) != length(max)
        throw(DimensionMismatch("number of rows in A $(size(A,1)) should be
        equal to length both vectors min and max: $min $max, respectively."))
    end

    sA = copy(A')
    for j in 1:length(min)
        sA[:, j] = unitScale(sA[:, j], min[j], max[j])
    end
    sA'
end

get_indicators() =
    let scale(f) = (X) -> begin
                            z = X .* -1
                            y = unitScale(z, -1 .* tpf_max[:], -1 .* tpf_min[:])
                            a = filter(b -> b > 1, y)
                            if length(a) > 0
                                @error a "DO NOT CONTINUE"
                                return;
                            end
                            f(y)
                        end
    # Independent Indicators
    independent_indicators = [
                            scale(hypervolumeIndicator),
                            # onvg,
                            # spacing,
                            # spread,
                            # scale(maximumSpread)
                            ]

    # Reference Indicators
    reference_indicators =  [
                            # onvgr,
                            # errorRatio,
                            # maxPFError,
                            # GD,
                            # IGD
                            ]

    independent_indicators = map(indicator -> ((R, A) -> indicator(A)), independent_indicators)
    indicators = vcat(independent_indicators, reference_indicators)
    indicators
end
indicators = get_indicators()
nindicators = length(indicators)

# Header to write in indicators files
header =  join(["Algorithm", "Run", "HV","ONVG", "SPACING", "SPREAD", "MAXIMUM SPREAD", "ONVGR", "Error Ratio", "Max Pareto Front Error", "GD", "IGD"], ',')
header


write_indicators(header, indicators) = let
    indicators_file = "$(output_folder)/indicators-HV-Max-$(Dates.format(Dates.now(), "yyyymmddHHMMSS")).csv"
    write(indicators_file, header) # Write header

    for a_i in 1:nalgorithms
        for r_i in nruns
            println("Analysing run $(r_i) from algorithm $(algorithms[a_i])")
            results = vcat(algorithms[a_i], r_i, [ind(tpfs[r_i], algorithm_pareto_results[a_i][r_i]) for ind in indicators])
            open(indicators_file, "a") do io
                write(io, "\n")
                join(io, results, ',')
            end
        end
    end
end

write_indicators(header, indicators)
