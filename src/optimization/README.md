# Optimization

  1. Sampling Techniques

    |  Applicable  |  Library   |
    |:------------:|:---------- |
    | Maybe        | [Ensemble](https://github.com/farr/Ensemble.jl): Implements in Julia various stochastic samplers. |
    | Maybe        | [GaussianRandomFields](https://github.com/PieterjanRobbe/GaussianRandomFields.jl): Julia Implementation of Gaussian Random Field Generation. |
    | **YES**      | [LatinHypercubeSampling](https://github.com/MrUrq/LatinHypercubeSampling.jl): Julia package for the creation of optimised Latin Hypercube Sampling Plans. |


  2. Dependencies Overview

    |  Applicable  |  Library   |
    |:------------:|:---------- |
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
    | Maybe       | [Mosek](https://github.com/JuliaOpt/Mosek.jl): MOSEK solves LP, SOCP, SDP, QP and MIP problems. Commercial software w/ availability for academic use. |
    | No          | [MovingWeightedLeastSquares](https://github.com/vutunganh/MovingWeightedLeastSquares.jl): Julia implementation of the moving weighted least squares method. Supports only v0.6. |
    | Maybe       | [MultiJuMP](https://github.com/anriseth/MultiJuMP.jl): Enables the user to easily run multi-objective optimization problems and generate Pareto Fronts. They use IpoptSolver. |
    | Maybe        | [NEOS](https://github.com/odow/NEOS.jl): Julia interface for NEOS Server, which is a free internet-based service for solving numerical optimization problems. License for some commercial solvers should be used solely for academic non-commercial research purposes. |
    | Probably not | [NLOptControl](https://github.com/JuliaMPC/NLOptControl.jl): Non linear control optimization tool. Requires usaing Ipopt or KNITRO. |
    | Maybe        | [NLSolversBase](https://github.com/JuliaNLSolvers/NLSolversBase.jl): Base package for optimization and equation solver software in JuliaNLSolvers. |
    | **Yes**      | [NLOpt](https://github.com/JuliaOpt/NLopt.jl): Wrapper to call the NLopt nonlinear-optimization library from the Julia language. |
    | No           | [NOMAD](https://github.com/jbrea/NOMAD.jl): Package to call the NOMAD blackbox optimization library from the Julia language. Supports only v.0.6. |
    | No           | [OSQP](https://github.com/oxfordcontrol/OSQP.jl): Julia wrapper for OSQP (Operator Splitting Quadratic Program Solver). |
    | Maybe        | [OptiMimi](https://github.com/jrising/OptiMimi.jl): Optimization for the Mimi.jl modeling framework. |
    | Maybe        | [Optim](https://github.com/JuliaNLSolvers/Optim.jl): Optimization functions for Julia. Univariate and multivariate optimization in Julia. |
    | No           | [OptimPack](https://github.com/emmt/OptimPack.jl): OptimPack.jl is the Julia interface to OptimPack, a library for solving large scale optimization problems. Does not support Julia 1.0. |
    | No           | [PARDISO](https://github.com/JuliaSparse/Pardiso.jl): Parallel Direct Solver wrapper in Julia. Requires license. |
    | No | [PiecewiseLinearOpt](https://github.com/joehuchette/PiecewiseLinearOpt.jl): A package for modeling optimization problems containing piecewise linear functions. Current support is for (the graphs of) continuous univariate functions. |
    | Maybe | [PolyJuMP](https://github.com/JuliaOpt/PolyJuMP.jl): JuMP extension for Polynomial Optimization. |
    | Probably Not | [ProximalAlgorithms](https://github.com/kul-forbes/ProximalAlgorithms.jl): Proximal algorithms for nonsmooth optimization in Julia. |
    | Maybe | [PolyJuMP](https://github.com/JuliaOpt/PolyJuMP.jl): JuMP extension for Polynomial Optimization. |
    | Maybe | [SCIP](https://github.com/SCIP-Interfaces/SCIP.jl): Julia Wrapper to SCIP (Solving Constraint Integer Programs). Addresses MIP, MINLP. |
    | Probably not | [SCS](https://github.com/JuliaOpt/SCS.jl): Julia Wrapper for SCS (Splitting Cone Solver). Addresses LP, SOCP, SDP, EXP, PCP. |  
    | Probably not | [SGDOptim](https://github.com/lindahua/SGDOptim.jl): A Julia package for Gradient Descent and Stochastic Gradient Descent. |
    | Probably not | [StructJuMP](https://github.com/StructJuMP/StructJuMP.jl): A block-structured optimization framework for JuMP. Provides a parallel algebraic modeling framework for block structured optimization models in Julia. StructJuMP, originally known as StochJuMP, is tailored to two-stage stochastic optimization problems and uses MPI to enable a parallel, distributed memory instantiation of the problem.|
    | Probably not | [Wallace](https://github.com/ChrisTimperley/Wallace.jl): High-performance evolutionary computation in Julia. Not available. |
    | Maybe | [Xpress](https://github.com/JuliaOpt/Xpress.jl):Julia interface for the FICO Xpress optimization suite. Including LP, QP, QCP, MILP, MIQP, and MIQCP. |
    | Maybe | [vOptGeneric](https://github.com/JuliaOpt/Xpress.jl): Solver of multiobjective linear optimization problems (MOCO, MOILP, MOMILP, MOLP): generic part. (Check [doc](https://github.com/vOptSolver/vOptSolver)) |
    | Maybe | [vOptSpecific](https://github.com/vOptSolver/vOptSpecific.jl): Solver of multiobjective linear optimization problems (MOCO, MOILP, MOMILP, MOLP): specific part. |

  3. Machine Learning Dependencies

    |  Applicable  |  Library   |
    |:------------:|:---------- |
    | **Yes**      | [BackpropNeuralNet](https://github.com/compressed/BackpropNeuralNet.jl): Julia implementation of a neural network. |
    | **Yes**      | [BayesNets](https://github.com/sisl/BayesNets.jl): Julia implementation of Bayesian Nets. |
    | **Yes**      | [Boltzmann](https://github.com/dfdx/Boltzmann.jl): Restricted Boltzmann machines and deep belief networks in Julia. |
    | **Yes**      | [CRF](https://github.com/slyrz/CRF.jl): Implementation of linear chain Conditional Random Fields implementation in Julia. |
    | **Yes**      | [Clustering](https://github.com/JuliaStats/Clustering.jl): Julia implementation of algorithms for data clustering. |
    | **Yes**      | [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl): Julia implementation of Decision Tree and Random Forest algorithms. |
    | Maybe        | [CombineML](https://github.com/ppalmes/CombineML.jl): Creates ensembles of machine learning models from scikit-learn, caret and Julia. |
    | Maybe        | [Flux](https://github.com/FluxML/Flux.jl): Flux is the ML library that doesn't make you tensor. |
    | Maybe        | [GaussianMixtures](https://github.com/davidavdav/GaussianMixtures.jl): Large scale Gaussian Mixture Models. |
    | Maybe        | [GaussianProcesses](https://github.com/STOR-i/GaussianProcesses.jl): Julia implementation of Gaussian Processes. |
    | Maybe        | [Isotonic](https://github.com/ajtulloch/Isotonic.jl): Isotonic Regression Algorithms in Julia. |
    | Maybe        | [KCores](https://github.com/johnybx/KCores.jl): A Julia package for k core decomposition. |
    | Maybe        | [KShiftsClustering](https://github.com/rened/KShiftsClustering.jl): Fast, low-memory method for batch and online clustering. |
    | Maybe        | [KernelDensity](https://github.com/JuliaStats/KernelDensity.jl): Kernel density estimators for Julia. |
    | Maybe        | [KernelDensityEstimate](https://github.com/JuliaRobotics/KernelDensityEstimate.jl): Kernel Density Estimate with product approximation using multiscale Gibbs sampling. |
    | Maybe        | [Klara](https://github.com/JuliaStats/Klara.jl): The Julia Klara package provides a generic engine for Markov Chain Monte Carlo (MCMC) inference. |
    | Maybe        | [Knet](https://github.com/denizyuret/Knet.jl): deep learning framework implemented in Julia. |
    | Maybe        | [Kriging](https://github.com/madsjulia/Kriging.jl): Gaussian process regressions and simulations. |
    | Maybe        | [LASSO](https://github.com/JuliaStats/Lasso.jl): Lasso.jl is a pure Julia implementation of the glmnet coordinate descent algorithm for fitting linear and generalized linear Lasso and Elastic Net models. |
    | No           | [LinearResponseVariationalBayes](https://github.com/rgiordan/LinearResponseVariationalBayes.jl): Julia tools for building simple variational Bayes models with JuMP. |
    | Maybe        | [LossFunctions](https://github.com/JuliaML/LossFunctions.jl): LossFunctions is a Julia package that provides efficient and well-tested implementations for a diverse set of loss functions that are commonly used in Machine Learning. |
    | Maybe        | [LowDimNearestNeighbors](https://github.com/yurivish/LowDimNearestNeighbors.jl): A lightweight implementation of nearest-neighbor search in low dimensions. |
    | Maybe        | [MLBase](https://github.com/JuliaStats/MLBase.jl): A set of functions to support the development of machine learning algorithms. Does not implement specific ML algorithms, instead it provides tools to support ML programs. |
    | No           | [MLDataPattern](https://github.com/JuliaML/MLDataPattern.jl): Utility package for subsetting, resampling, iteration, and partitioning of various types of data sets in Machine Learning. Tests are failing. |
    | No           | [MLDataUtils](https://github.com/JuliaML/MLDataUtils.jl): Utility package for generating, loading, partitioning, and processing Machine Learning datasets. Supports v.1.0.|
    | No           | [MLDatasets](https://github.com/JuliaML/MLDatasets.jl): provide a common interface for accessing common Machine Learning (ML) datasets. Focused on downloading, unpacking, and accessing benchmark dataset. |
    | Maybe        | [MLKernels](https://github.com/trthatcher/MLKernels.jl): Julia package for Mercer kernel functions (or the covariance functions used in Gaussian processes) that are used in the kernel methods of machine learning. This package provides a flexible datatype for representing and constructing machine learning kernels as well as an efficient set of methods to compute or approximate kernel matrices. |
    | No           | [MLLabelUtils](https://github.com/JuliaML/MLLabelUtils.jl): Utility package for working with classification targets. As such, this package provides the necessary functionality for interpreting class-predictions, as well as converting classification targets from one encoding to another. |
    | Maybe        | [MXNet](https://github.com/dmlc/MXNet.jl): Julia package provide flexible and efficient deep learning in Julia. |
    | No        | [MachineLearning](https://github.com/benhamner/MachineLearning.jl): Machine Learning Library in JUlia. Is not being maintained. |
    | Maybe        | [Mamba](https://github.com/brian-j-smith/Mamba.jl): Markov Chain Monte Carlo (MCMC) for Bayesian Analysis in Julia. |
    | Maybe        | [ManifoldLearning](https://github.com/wildart/ManifoldLearning.jl): A Julia package for manifold learning and nonlinear dimensionality reduction. |
    | Maybe        | [Merlin](https://github.com/hshindo/Merlin.jl): Deep learning for Julia. It aims to provide a fast, flexible and compact deep learning library for machine learning. Merlin is tested against Julia 1.0 on Linux, OS X, and Windows (x64). |
    | Maybe        | [Mocha](https://github.com/pluskid/Mocha.jl): Deep learning framework for Julia (inspired by Caffe). |
    | Probably not | [NaiveBayes](https://github.com/dfdx/NaiveBayes.jl): Naive bayes classifier. |
    | Maybe        | [NearestNeighbors](https://github.com/KristofferC/NearestNeighbors.jl): High performance nearest neighbor data structures and algorithms for Julia. |
    | No        | [PLSRegressor](https://github.com/lalvim/PLSRegressor.jl): Partial Least Squares Regressor. Supports v0.6. Julia. |
    | Maybe        | [Perceptron](https://github.com/lalvim/Perceptrons.jl): Set of perceptron algorithms. |
    | Maybe        | [QuickShiftClustering](https://github.com/rened/QuickShiftClustering.jl): Fast hierarchical medoid clustering. |
    | Maybe        | [Regression](https://github.com/lindahua/Regression.jl): Algorithms for regression analysis (e.g. linear regression and logistic regression). |
    | Maybe        | [Salsa](https://github.com/jumutc/SALSA.jl): Software Lab for Advanced Machine Learning with Stochastic Algorithms in Julia. |
    | Maybe        | [SOM](https://github.com/LiScI-Lab/SOM.jl): Kohonen's self-organising maps for Julia. |
    | Maybe        | [SVR](https://github.com/madsjulia/SVR.jl): Support Vector Regression using libSVM. |
    | Maybe        | [ScikitLearn](https://github.com/cstjean/ScikitLearn.jl): Julia implementation of the scikit-learn API. |
    | Maybe        | [SimilaritySearch](https://github.com/sadit/SimilaritySearch.jl): A Near Neighbor Search Library. |
    | Maybe        | [Smile](https://github.com/sisl/Smile.jl): A Julia wrapper for the Smile C++ Structural Modeling, Inference, and Learning Engine for Bayesian & Influence Networks. |
    | Maybe        | [SmoothingKernels](https://github.com/johnmyleswhite/SmoothingKernels.jl): Smoothing kernels for use in kernel regression and kernel density estimation. |
    | Maybe        | [SmoothingSplines](https://github.com/nignatiadis/SmoothingSplines.jl): Cubic smoothing splines in Julia. |
    | Maybe        | [SparseRegression](https://github.com/joshday/SparseRegression.jl): Statistical Models with Regularization in Pure Julia. |
    | Maybe        | [SubsetSelection](https://github.com/jeanpauphilet/SubsetSelection.jl): Fast Subset Selection algorithm for Statistics/Machine Learning. |
    | Maybe        | [TSne](https://github.com/lejon/TSne.jl): Julia port of L.J.P. van der Maaten and G.E. Hintons T-SNE visualisation technique. |
    | Maybe        | [TensorFlow](https://github.com/malmaud/TensorFlow.jl): A wrapper around TensorFlow, a popular open source machine learning framework from Google. |
    | Maybe        | [XGBoost](https://github.com/dmlc/XGBoost.jl): XGBoost (eXtreme Gradient Boosting) Julia Package. |
    | No           | [XGrad](https://github.com/dfdx/XGrad.jl): XGrad is a package for finding gradients of functions and symbolic expressions. |
    | Maybe           | [kNN](https://github.com/johnmyleswhite/kNN.jl): The k-nearest neighbors algorithm in Julia. |




   4. Utilities.

       |  Applicable  |  Library   |
       |:------------:|:---------- |
       | Maybe        | [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl): Rigorous floating point calculations using interval arithmetic in Julia. |
       | Maybe        | [IntervalConstraintProgramming](https://github.com/JuliaIntervals/IntervalConstraintProgramming.jl): Calculate rigorously the feasible region for a set of real-valued inequalities with Julia. |
       | Maybe        | [IntervalRootFinding](https://github.com/JuliaIntervals/IntervalRootFinding.jl): Find all roots of a function in a guaranteed way with Julia. |
       | Maybe        | [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl): Data structures for Optimization models. |
       | Maybe        | [NormalizeQuantiles](https://github.com/oheil/NormalizeQuantiles.jl): Julia package for quantile normalization. |
