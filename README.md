# Algorithmic Design OPTimization (ADOPT)




*Disclaimer*
This is the result of a work in progress optimization tool that is currently
being maintained by the [Algorithmic Design and Analysis (ADA) group](https://algorithmicdesign.github.io/).  

## 30 Seconds to ADOPT

This experiment must be run in a Julia environment (with Julia 1.1.0).


## 2. Getting Started

In its current version, ADOPT depends on Julia 1.+ and a few Python libraries, particularly:
- [Platypus](https://github.com/Project-Platypus/Platypus)
- [Sklearn](https://scikit-learn.org/stable/)


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
in your Python environment and would like to re-use them then consider [2.1.1. Re-use existing Python frameworks](https://github.com/PastelBelem8/ADOPT.jl/#2.1.1.Re-useexistingPythonframeworks), else just skip
to [2.2. Installing ADOPT](https://github.com/PastelBelem8/ADOPT.jl/###2.2.InstallingAdopt).

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

#### 2. Installing ADOPT

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
-
