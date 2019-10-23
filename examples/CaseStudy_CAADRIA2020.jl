using DelimitedFiles
objs_cols = collect(10:11) # Objectives
has_header = true

filepath = "./examples/SMPSO_results_run2.csv"

read_cols(filename, dlm=','; objs=(:), header=false) = open(filename, "r") do io
    if header readline(io); end
    readdlm(io, dlm, Float64, '\n')[:, objs]
end



A = read_cols(filepath, objs=objs_cols, header=has_header)
A_min = mapslices(minimum, A, dims=1)[:]
A_max = mapslices(maximum, A, dims=1)[:]

indicators = Dict(:hv => Main.ADOPT.hypervolumeIndicator)
results = Main.ADOPT.compute_per_iteration(A', 15, indicators, [:MAX, :MIN], A_min, A_max)
