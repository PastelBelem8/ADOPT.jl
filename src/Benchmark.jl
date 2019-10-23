#=
    This file introduces the mechanisms to allow the automatic execution of
    different optimization runs, either for:
    1 - Test an algorithm in different problems
    2 - Compare the performance of different algorithms accross one or more problems
    3 - Test the effect of different parameters
=#

#=
A benchmark plan encloses all the combinations to be executed.
A benchmark plan will have the following structure
    (id, algorithm, params)+
=#
create_benchmark_plan(ids, as, ps) =
    map((i, a, p) -> "[ID-$i] = $a $p", ids, as, ps)

init_benchmark(bdir, bplan=nothing) =
    with(results_dir, bdir, file_sep, "\n") do
        @info "[$(now())][Benchmark][$init_benchmark] Create benchmarkdir $(bdir)"
        mkdir(results_dir())
        isnothing(bplan) ? nothing : write_content("benchmark", "$(bdir)/benchmark.plan", bplan)
    end

log_error(err_file, e) =
    write_content("error", err_file, ["\n>>>>> Error <<<<<\n",
    sprint(showerror, e, catch_backtrace())])

get_algorithm(config::Tuple) = first(config)
get_algorithm(config) = config

get_params(config::Tuple) = length(config) == 2 ? config[2] : Dict()
get_params(config::T) where{T<:AbstractSolver} = Dict() # Assumes all configurations have been set

"""
    benchmark(;[bname,] nruns, [(alg1, params1), (alg2,), ...], problem, max_evals)
    benchmark(;nruns, [(alg1, params1), (alg2,), ...], problem, max_evals)
    benchmark(;nruns, [(alg1, params1), Solver, ...], problem, max_evals)
    benchmark(;nruns, [alg1, (alg2, params1), Solver, ...], problem, max_evals)

    Sequentially runs all algorithms `alg_i` with the configuration provided
    in params1. All parameters that are not passed in `params_i` are assumed
    to take the default value.

    To verify their default value, please consult the documentation of the
    referred libraries (e.g., Platypus for MOEAs, ScikitLearn for ML methods).
"""
benchmark(;bname="benchmark", nruns, algorithms, problem, max_evals=100) = let
    BENCHMARK_ID = "$(results_dir())/$(bname)-$(get_unique_string())"

    alg_ids = 1:length(algorithms)
    algs = map(get_algorithm, algorithms)
    algs_params = map(get_params, algorithms)
    plan = create_benchmark_plan(alg_ids, algs, algs_params)

    init_benchmark(BENCHMARK_ID, plan);

    for id in alg_ids
        @info "[$(now())][benchmark] Starting $nruns for algorithm $(algs[id]) (with id $id) with params: $(algs_params[id])."
        with(results_dir, "$(BENCHMARK_ID)/$(id)") do
            failsafe_mkdir("benchmark", results_dir())
            for run in 1:nruns
                try
                    @info "[$(now())][benchmark] Starting run $(run) (out of $(nruns))."
                    solve(algs[id], algs_params[id], problem, max_evals, true)
                catch e
                    @error "[$(now())][benchmark] Error $(sprint(show, e))\nCheck error.log file for more information.."
                    log_error("$(BENCHMARK_ID)/error.log", e) end
            end
        end
    end
end

# Especially designed for Robot Case study. Cuz I need each solver to be created all over again, to prevent it from storing
solvers_benchmark(;bname="benchmark", nruns, solvers, problem, max_evals=100) = let
    BENCHMARK_ID = "$(results_dir())/$(bname)-$(get_unique_string())"

    alg_ids = 1:length(solvers)
    init_benchmark(BENCHMARK_ID);

    for id in alg_ids
        @info "[$(now())][benchmark] Starting $nruns for algorithm $(solvers[id]) (with id $id) with params."
        with(results_dir, "$(BENCHMARK_ID)/$(id)") do
            failsafe_mkdir("benchmark", results_dir())
            for run in 1:nruns
                try
                    @info "[$(now())][benchmark] Starting run $(run) (out of $(nruns))."
                    solve(solvers[id](), Dict(), problem, max_evals, true)
                catch e
                    @error "[$(now())][benchmark] Error $(sprint(show, e))\nCheck error.log file for more information.."
                    log_error("$(BENCHMARK_ID)/error.log", e) end
            end
        end
    end
end



export benchmark
