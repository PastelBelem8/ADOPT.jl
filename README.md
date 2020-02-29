# Algorithmic Design OPTimization (ADOPT)

ADOPT is a Julia optimization tool that interfaces with multiple Python frameworks,
including Derivative-Free Optimization ones (e.g., _platypus_ and _nlopt_), and
also data science ones (e.g., _scikit-learn_) to support the resolution of both
Single- and Multi-objective Optimization problems.


## 1. 30 Seconds to ADOPT

## 1.1. Quick Start
Consider the following example, whose goal is to minimize two analytical functions
`f1` and `f2`, which only depends on a single continuous variable, whose values
may range in the interval $[-10, 10]$.
```julia
using ADOPT
# Analytical Objective Functions
f1(x) = x^2
f2(x) = (x-2)^2

let var1 = RealVariable(-10, 10),
    obj1 = Objective(f1, :MIN),
    obj2 = Objective(f2, :MIN),
    model = Model([var1], [obj1, obj2])
  solve(NSGAII, model, max_evals=100)
end
```

## 1.2. Quick

```julia
using ADOPT
using ADOPT.Platypus
using ADOPT.Sampling
using ADOPT.ScikitLearnModels

# Schaffer function nº 1 (Unconstrained)
schaffer1_vars = [RealVariable(-10, 10)]

schaffer1_f1(x) = x[1]^2
schaffer1_f2(x) = (x[1] - 2)^2
schaffer1_objs = [Objective(schaffer1_f1), Objective(schaffer1_f2)]
schaffer1 = Model(schaffer1_vars, schaffer1_objs)

# Experiment two Algorithms
algorithms_to_test = [
    (NSGAII,),  # default params
    (NSGAII, Dict(:population_size => 15)), (randomMC, Dict(:nsamples=> 500))
]

benchmark(
    # Run 3 times the optimization algorithm
    nruns = 3,
    # Vector of algorithms to test
    algorithms = algorithms_to_test,
    # problem to solve
    problem = schaffer1,
    # maximum number of function evaluations to make for each objective function
    max_evals = 100)


# Test 2. Use Solvers and Algorithms
ADOPT.ScikitLearnModels.@sk_import gaussian_process: (GaussianProcessRegressor,)

surrogate = Surrogate(  GaussianProcessRegressor(),
                        objectives=schaffer1_objs,
                        creation_f=sk_fit!,
                        update_f=sk_fit!,
                        evaluation_f=sk_predict)

meta_params = Dict(:sampling_function => randomMC, :nsamples => 30)
optimiser2 = ADOPT.PlatypusSolver(NSGAII, max_eval=500, algorithm_params=Dict(:population_size => 50), nondominated_only=true)
solver = ADOPT.MetaSolver(optimiser2; surrogates=[surrogate], max_eval=200, sampling_params=meta_params, nondominated_only=true)
benchmark(nruns=1, algorithms=[solver, (NSGAII,), (NSGAII, Dict(:population_size => 15)), (randomMC, Dict(:nsamples=> 500))], problem=schaffer1, max_evals=100)

```

## 2. Getting Started
This experiment must be run in a Julia environment (with Julia 1.1.0).

ADOPT is a
Julia optimization tool that

### 2.1. Installing the Pre-requisites

This section focus on the installation and verification that the necessary dependencies
to run ADOPT are satisfied. The first one is to have a **Julia** executable. In case,
you have not downloaded Julia yet, you may refer to the [Official Julia Webpage](https://julialang.org/downloads/).

Open up a command line and execute the following command to verify your Julia version.
```
$ julia --version
```

The second dependency is **Python3**. Consider the [Python Official Documentation](https://www.python.org/downloads/)
if you haven't installed Python yet. Verify that you have Python installed by runnign the following command in the
command line:

```
$ python --version
```

ADOPT makes use of two different Python libraries: *sklearn* and *platypus*. In order to
work, you have to install both libraries. If you already have both frameworks installed
in your Python environment and would like to re-use them then consider [2.1.1. Re-use existing Python frameworks](https://github.com/PastelBelem8/ADOPT.jl/#211-re-use-existing-python-frameworks), else just skip
to [2.2. Installing ADOPT](https://github.com/PastelBelem8/ADOPT.jl/#22-installing-adopt).

#### 2.1.1. Re-use existing Python frameworks

In order to integrate with Python frameworks, ADOPT makes use of `PyCall.jl`, which
by default will create a self-contained environment within the Julia installation.
However, it is often the case that we would like to re-use some Python environment
rather than having Julia creating its own (e.g., space constraints, local changes).

In that case, if you already have a ready to use Python environment, you may
consider reconstructing PyCall with respect to that environment, instead of having
it creating a new environment.

```julia
julia> ENV["PYTHON"] = "<path/to/python/executable>"
julia> using Pkg
julia> Pkd.add("PyCall")
```

### 2.2. Installing ADOPT

To install ADOPT, open up a Julia terminal, enter the `pkg` mode by typing _]_
in the Julia terminal, and execute the following instructions:

```
julia> add https://github.com/PastelBelem8/ADOPT.jl
```

Verify that ADOPT was successfully installed by running the following
instructions in a Julia terminal:

```
julia> using ADOPT
```

# Contributors

- @PastelBelem8
- @ines-pereira


*Disclaimer*

This is the result of a work in progress optimization tool that is currently
being maintained by the [Algorithmic Design and Analysis (ADA) group](https://algorithmicdesign.github.io/).  
