# Explainability

1. Sensitivity Analysis (SA)

  Sensitivity Analysis can be used to provide feedback about how much each variable contributes for the variation of the output value.


    |   Applicable   |   Library   |
    |: ------------ :|: ---------- |
    | **Yes**        | [BIGUQ](https://github.com/madsjulia/BIGUQ.jl): Performs Bayesian Information Gap Decision Theory (BIG-DT) analysis for Uncertainty Quantification, Experimental Design and Decision Analysis. |
    | **Yes**        | [BayesianNonparametrics](https://github.com/OFAI/BayesianNonparametrics.jl): Implementation of state-of-the-art Bayesian nonparametric models for medium-sized unsupervised problems. Useful for explaining discrete or continuous. |
    | Maybe          | [GeneralizedMetropolisHastings](https://github.com/QuantifyingUncertainty/GeneralizedMetropolisHastings.jl): Parallel MCMC implementation, the core package of the Quantifying Uncertainty project. |
    | No         | [GeneralizedSampling](https://github.com/robertdj/GeneralizedSampling.jl): Has no relevant sampling methods. |
    | Maybe        | [HypothesisTests](https://github.com/JuliaStats/HypothesisTests.jl): Implement a wide range of hypothesis tests. 


2.


3. Variables Correlation


    |   Applicable   |   Library   |
    |: ------------ :|: ---------- |
    | Maybe          | [CovarianceMatrices](https://github.com/gragusa/CovarianceMatrices.jl): Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation for Julia. |
    | Maybe          | [GARCH](https://github.com/AndreyKolev/GARCH.jl): Generalized Autoregressive Conditional Heteroskedastic (GARCH) models for Julia. |
