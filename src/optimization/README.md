# Optimization

1. Sampling Techniques

  |   Applicable   |   Library   |
  |: ------------ :|: ---------- |
  | Maybe          | [Ensemble](https://github.com/farr/Ensemble.jl): Implements in Julia various stochastic samplers. |
  | Maybe | [GaussianRandomFields](https://github.com/PieterjanRobbe/GaussianRandomFields.jl): Julia Implementation of Gaussian Random Field Generation. |
  | **YES** | [LatinHypercubeSampling](https://github.com/MrUrq/LatinHypercubeSampling.jl): Julia package for the creation of optimised Latin Hypercube Sampling Plans. |


2. Dependencies Overview

  |   Applicable   |   Library   |
  |: ------------ :|: ---------- |
  | Maybe            | [AbstractOperators](https://github.com/kul-forbes/AbstractOperators.jl): Abstract operators extend the syntax typically used for arbitrary dimensions and nonlinear functions. Abstract operators apply the mappings with specific efficient algorithms that minimize memory requirements. Useful for iterative algorithms and in first order large-scale optimization algorithms. |
  | Maybe | [BARON](https://github.com/joehuchette/BARON.jl): Nonconvex global mixed-integer optimization problems. |
  | **Yes** | [BlackBoxOptim](https://github.com/robertfeldt/BlackBoxOptim.jl): Multi- and Single Objective optimization problems w/ focus on (meta-)heuristics. (Verify which algorithms it contains!) |
  | No | [CPLEX](https://github.com/JuliaOpt/CPLEX.jl): Julia interface for the CPLEX optimization software. Requires a license. |
  | Maybe | [CUTEst](https://github.com/JuliaSmoothOptimizers/CUTEst.jl): Constrained and unconstrained nonlinear programming problems for testing and comparing optimization algorithms. Derived from NLP models. |
  | Maybe | [CVXOPT](https://github.com/cvxopt/CVXOPT.jl): Julia interface to CVXOPT ([Python Software for Convex Optimization](http://cvxopt.org/)). |
  | Maybe | [CBC](https://github.com/JuliaOpt/Cbc.jl): Julia interface to the MILP programming solver Cbc. |
  | Maybe | [Clp](https://github.com/JuliaOpt/Clp.jl): Julia interface to the Coin-OR Linear Programming solver (CLP). |
  | No    | [ConicNonlinearBridge](https://github.com/mlubin/ConicNonlinearBridge.jl): MathProgBase wrapper to solve conic optimization problems with derivative based nonlinear solvers. |
  | No    | [CutPruners](https://github.com/JuliaPolyhedra/CutPruners.jl): Pruning algorithms. |
  | No    | [DReal](https://github.com/dreal/DReal.jl): Applies to satisfiability problems. |
  | No | [CoinOptServices](https://github.com/JuliaOpt/CoinOptServices.jl): Julia interface to [COIN-OR Optimization Services](https://projects.coin-or.org/OS). Nonlinear objective and constraint functions. Builds failing and has problems in Linux and macOS. |
  | No | [Parajito](https://github.com/JuliaOpt/Pajarito.jl): Solver for mixed-integer convex optimization (MICP). Does not support Julia 1.0. |
  | No | [Katana](https://github.com/lanl-ansi/Katana.jl): Cutting plane based solver for Convex NLPs. Builds fail. |
  | Maybe | [Pod](https://github.com/lanl-ansi/POD.jl): Global solver for Nonconvex MINLPs, MIQCP and NLP. Support v0.6.4. |
  | Maybe | [Juniper](https://github.com/lanl-ansi/Juniper.jl): JuMP-based Nonlinear Integer Program Solver (MINLP). Heuristic for non convex problems. If you need the gloval optimum check Pod.jl. Support v0.6. |
  | Maybe | [EAGO](https://github.com/PSORLab/EAGO.jl): Easy Advanced Global optimization. |
  | Maybe | [EAGOBranchBound](https://github.com/PSORLab/EAGOBranchBound.jl): Branch and Bound library for Julia. |
  | Maybe | [EAGODomainReduction](https://github.com/MatthewStuber/EAGODomainReduction.jl): Package for domain reduction in Global Optimization. |
  | Maybe | [EAGOSmoothMcCormickGrad](https://github.com/MatthewStuber/EAGOSmoothMcCormickGrad.jl): A (differentiable) McCormick Relaxation Library w/Embedded (sub)gradient. |
  | No | [ECOS](https://github.com/JuliaOpt/ECOS.jl): Wrapper for the ECOS (A lightweight conic solver for second-order cone programming) embeddable conic optimization interior point solver. Does not support v1.0. |
  | **YES** | [Evolutionary](https://github.com/wildart/Evolutionary.jl): Evolutionary and Genetic Algorithms for Julia. CMA-ES, SA-ES, GAs. |
  | Maybe | [FirstOrderSolvers](https://github.com/mfalt/FirstOrderSolvers.jl): Large scale convex optimization solvers in Julia. |
  | **YES** | [GAFramework](https://github.com/vvjn/GAFramework.jl): Genetic Algorithm with multi-threading. |
  | Maybe   | [GLM](https://github.com/JuliaStats/GLM.jl): Generalized linear models in Julia. |
  | Maybe   | [GLMNet](): Julia wrapper for fitting Lasso/ElasticNet GLM models using glmnet. |
  | No      | [GLPK](): Julia wrapper for the GNU Linear Programming Kit Library. |
  | Maybe   | [GeneticAlgorithms](https://github.com/WestleyArgentum/GeneticAlgorithms.jl): Framework for creating genetic algorithms and running them in parallel. |
  | No   | [Gurobi](https://github.com/JuliaOpt/Gurobi.jl): Commercial optimization solver for a variety of mathematical programming problems, including LP, QP, QCP, MILP, MIQP, and MIQCP. |
  | Maybe          | [IntervalOptimization](https://github.com/JuliaIntervals/IntervalOptimisation.jl): Rigorous global optimisation in Julia. Last build failed. |
  | Maybe          | [Ipopt](https://github.com/JuliaOpt/Ipopt.jl): Julia interface to the Ipopt nonlinear solver. |
  | Maybe          | [IterativeSolvers](https://github.com/JuliaMath/IterativeSolvers.jl): Iterative algorithms for solving linear systems, eigensystems, and singular value problems. |
  | Maybe          | [JuMP](https://github.com/JuliaOpt/JuMP.jl): Modeling language for Mathematical Optimization (linear, mixed-integer, conic, semidefinite, nonlinear). |
  | Maybe          | [JuMPChance](https://github.com/mlubin/JuMPChance.jl): A JuMP extension for probabilistic (chance) constraints. |
  | Maybe          | [JuMPeR](https://github.com/IainNZ/JuMPeR.jl): Julia for Mathematical Programming - extension for Robust Optimization. |
  | No          | [KNITRO](https://github.com/JuliaOpt/KNITRO.jl): Requires license. |
  | Maybe       | [LBFGS](https://github.com/Gnimuc/LBFGSB.jl): Julia wrapper for L-BFGS-B Nonlinear Optimization Code. |
  | Maybe       | [LearningStrategies](https://github.com/JuliaML/LearningStrategies.jl): LearningStrategies is a modular framework for building iterative algorithms in Julia. |
  | Maybe       | [LeastSquaresOptim](https://github.com/matthieugomez/LeastSquaresOptim.jl): Dense and Sparse Least Squares Optimization. |
  | Maybe       | [LineSearches](https://github.com/JuliaNLSolvers/LineSearches.jl): Line search methods for optimization and root-finding. |
  | Maybe       | [LinearLeastSquares](https://github.com/davidlizeng/LinearLeastSquares.jl): Least squares solver in Julia. |
  |


3. Machine Learning Dependencies

  |   Applicable   |   Library   |
  |: ------------ :|: ---------- |
  | **Yes**        | [BackpropNeuralNet](https://github.com/compressed/BackpropNeuralNet.jl): Julia implementation of a neural network. |
  | **Yes**        | [BayesNets](https://github.com/sisl/BayesNets.jl): Julia implementation of Bayesian Nets. |
  | **Yes**        | [Boltzmann](https://github.com/dfdx/Boltzmann.jl): Restricted Boltzmann machines and deep belief networks in Julia. |
  | **Yes**        | [CRF](https://github.com/slyrz/CRF.jl): Implementation of linear chain Conditional Random Fields implementation in Julia. |
  | **Yes**        | [Clustering](https://github.com/JuliaStats/Clustering.jl): Julia implementation of algorithms for data clustering. |
  | **Yes**        | [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl): Julia implementation of Decision Tree and Random Forest algorithms. |
  | Maybe          | [CombineML](https://github.com/ppalmes/CombineML.jl): Creates ensembles of machine learning models from scikit-learn, caret and Julia. |
  | Maybe | [Flux](https://github.com/FluxML/Flux.jl): Flux is the ML library that doesn't make you tensor. |
  | Maybe | [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl): Large scale Gaussian Mixture Models. |
  | Maybe | [GaussianProcesses](https://github.com/STOR-i/GaussianProcesses.jl): Julia implementation of Gaussian Processes. |
  | Maybe | [Isotonic](https://github.com/ajtulloch/Isotonic.jl): Isotonic Regression Algorithms in Julia. |
  | Maybe | [KCores](https://github.com/johnybx/KCores.jl): A Julia package for k core decomposition. |
  | Maybe | [KShiftsClustering](https://github.com/rened/KShiftsClustering.jl): Fast, low-memory method for batch and online clustering. |
  | Maybe  | [KernelDensity](https://github.com/JuliaStats/KernelDensity.jl): Kernel density estimators for Julia. |
  | Maybe  | [KernelDensityEstimate](https://github.com/JuliaRobotics/KernelDensityEstimate.jl): Kernel Density Estimate with product approximation using multiscale Gibbs sampling. |
  | Maybe | [Klara](https://github.com/JuliaStats/Klara.jl): The Julia Klara package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference. |
  | Maybe  | [Knet](https://github.com/denizyuret/Knet.jl): deep learning framework implemented in Julia. |
  | Maybe  | [Kriging](https://github.com/madsjulia/Kriging.jl): Gaussian process regressions and simulations. |
  | Maybe       | [LASSO](https://github.com/JuliaStats/Lasso.jl): Lasso.jl is a pure Julia implementation of the glmnet coordinate descent algorithm for fitting linear and generalized linear Lasso and Elastic Net models. |
  | No  | [LinearResponseVariationalBayes](https://github.com/rgiordan/LinearResponseVariationalBayes.jl): Julia tools for building simple variational Bayes models with JuMP. |
  | Maybe   | [LossFunctions](https://github.com/JuliaML/LossFunctions.jl): LossFunctions is a Julia package that provides efficient and well-tested implementations for a diverse set of loss functions that are commonly used in Machine Learning. |
  | Maybe   | [LowDimNearestNeighbors](https://github.com/yurivish/LowDimNearestNeighbors.jl): A lightweight implementation of nearest-neighbor search in low dimensions. |
  | 

 Utilities.

 |   Applicable   |   Library   |
 |: ------------ :|: ---------- |
 | Maybe          | [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl): Rigorous floating point calculations using interval arithmetic in Julia. |
 | Maybe          | [IntervalConstraintProgramming](https://github.com/JuliaIntervals/IntervalConstraintProgramming.jl): Calculate rigorously the feasible region for a set of real-valued inequalities with Julia. |
 | Maybe          | [IntervalRootFinding](https://github.com/JuliaIntervals/IntervalRootFinding.jl): Find all roots of a function in a guaranteed way with Julia. |
 |
