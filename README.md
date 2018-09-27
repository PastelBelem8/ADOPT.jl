# Msc-thesis
This is the result of my master thesis on Multi-Objective Optimization. This repository is more focused towards Pareto-based optimization rather than Single-Objective optimization with preference articulation. We focus on time-consuming optimization routines and, as a result we focus on model-based methods to allow for faster convergence times. This is relevant for Architectural Design Optimization, which depends on time-intensive simulations (e.g. minutes, hours or even days to complete a single simulation).

| **Package Status** | **Build Status**  |
|:------------------:|:-----------------:|
| [![License](https://img.shields.io/badge/license-GNU-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-missing-red.svg)]() | [![Build Status](https://travis-ci.com/PastelBelem8/Msc-thesis.svg?token=tFNrx3GDmxzsVPAGpUzX&branch=master)](https://travis-ci.com/PastelBelem8/Msc-thesis) |

|    Optimization    |    Metrics    |    Benchmarks/tests    |    Explainability    |    Graphical User Interface    |
| ------------------ | ------------- | ---------------------- | -------------------- | ------------------------------ |
| Single/Multi       | Pareto Based (Spread, GD, IDG, HVI, Spacing, EpsilonIndicator) | CEC2009                | Sensitivity Analysis | Constraints Preview            |
| Multi/Threaded     | Accuracy, Precision, Improvement (model-based quality metrics)               | DLTZ                   | (Screening, etc.)    | Feedback of process evolution  |
| Events             |               | WFG                    | LRP                  | Representation of Pareto Front |
| Integer/Continuous |               | ZDT                    | Taylor Decomp ?      | Accessibility                  |
| (un)Constrained    |               | GTOC, FDA              | Rule Based ?         | Variable Importance            |
| Dynamic changes    |               | Viennet, Schaffer      | Parse Comments ?     | Tips for better setup of algs. |


Crazy Ideas to be addressed in thesis:
- Will previously generalized models for standardized tests be less or more performant on a case study in Architectural Design Optimization (ADO)?
- Will I be able to properly represent the Pareto-based and the plots of evolution?
- Will I be able to extend [Khepri](https://github.com/aptmcl/Khepri) w/ an EP backend?
- Will I be able to provide an implementation different from [ParEGO](https://www.cs.bham.ac.uk/~jdk/parego/ParEGO-TR3.pdf) that uses model-based methods?

Plans:
- Do a benchmark in MOO for different case studies in ADO.
- Do a comparative study between different model-based approaches (multiple surrogates __vs__ single surrogate)

A more objective description of each module and the goals of this thesis can be found in The [Projects Section](https://github.com/PastelBelem8/Msc-thesis/projects).
