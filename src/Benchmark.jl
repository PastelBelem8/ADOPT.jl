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


init_benchmark(bdir, bplan) =
    with(results_dir, bdir, file_sep, "\n") do
        @info "[$(now())][Benchmark][$init_benchmark] Create benchmarkdir $(bdir)"
        mkdir(bdir)
        write_content("benchmark", "$(bdir)/benchmark.plan", bplan)
    end

log_error(err_file, e) =
    write_content("error", err_file, ["\n>>>>> Error <<<<<\n",
    sprint(showerror, e, catch_backtrace())])


"""
    benchmark(;[bname,] nruns, [(alg1, params1), (alg2,), ...], problem, max_evals)
    benchmark(;nruns, [(alg1, params1), (alg2,), ...], problem, max_evals)

    Sequentially runs all algorithms `alg_i` with the configuration provided
    in params1. All parameters that are not passed in `params_i` are assumed
    to take the default value.

    To verify their default value, please consult the documentation of the
    referred libraries (e.g., Platypus for MOEAs, ScikitLearn for ML methods).
"""
benchmark(;bname="benchmark", nruns, algorithms, problem, max_evals) = let
    BENCHMARK_ID = "$(results_dir())/$(bname)-$(get_unique_string())"

    alg_ids = 1:length(algorithms)
    algs = map(first, algorithms)
    algs_params = map(t -> length(t) == 2 ? t[2] : Dict(), algorithms)
    plan = create_benchmark_plan(alg_ids, algs, algs_params)

    init_benchmark(BENCHMARK_ID, plan);

    for id in alg_ids
        @info "[$(now())][benchmark] Starting $nruns for algorithm $(algs[id]) (with id $id) with params: $(algs_params[id])."
        with(results_dir, "$(BENCHMARK_ID)/$(id)") do
            failsafe_mkdir("benchmark", results_dir())
            for run in 1:nruns
                try
                    @info "[$(now())][benchmark] Starting run $(run) (out of $(nruns))."
                    solve(algorithm=algs[id], params=algs_params[id], max_evals=max_evals, problem=problem)
                catch e
                    @error "[$(now())][benchmark] Error $(sprint(show, e))\nCheck error.log file for more information.."
                    log_error("$(BENCHMARK_ID)/error.log", e) end
            end
        end
    end
end

export benchmark
