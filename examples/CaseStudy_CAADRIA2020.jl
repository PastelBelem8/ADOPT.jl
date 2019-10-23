using DelimitedFiles
objs_cols = collect(10:11) # Objectives
has_header = true

filepath = "./examples/SMPSO_results_run2.csv"

read_cols(filename, dlm=','; objs=(:), header=false) = open(filename, "r") do io
    if header readline(io); end
    readdlm(io, dlm, Float64, '\n')[:, objs]
end



# A = read_cols(filepath, objs=objs_cols, header=has_header)
A = [1 1]


A_min = [0, 0] # mapslices(minimum, A, dims=1)[:]
A_max = [2, 2] # mapslices(maximum, A, dims=1)[:]
# this is throwing an error... why?
Main.ADOPT.hypervolume(A', [:MAX, :MAX], A_min, A_max)



indicators = Dict(:hv => x -> Main.ADOPT.hypervolume(x, [:MIN, :MIN], A_min, A_max))
results = Main.ADOPT.compute_per_iteration(A, indicators, 225)
