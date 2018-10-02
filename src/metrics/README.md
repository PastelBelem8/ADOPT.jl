# Metrics

Multi-Objective Optimization problems (MOO) belong to the set of problems concerned with the optimization of more than one objective simultaneously. The addition of other, potentially conflicting, objectives to the optimization process prompts the re-definition of optimality. While in Single-Objective problems we expect the optimal solution to be the set of parameters that achieve the best objective value possible, in multi-objective the best possible configuration for one of the objectives is rarely the best possible configuration for all other objectives as well. This may be explained by the fact that the objectives are often contradictory.

In order to be able to compare different multi-objective solutions, these problems are often addressed considering the Pareto optimality (or Pareto Efficiency) concept [1]. This concept, named after the economist Vilfredo Pareto, defines an optimal solution as being a solution for which it is impossible to improve an objective value without deteriorating others. Such a solution is also said to be non-dominated or noninferior. The set of non-dominated solutions is called the Pareto Front (or Pareto Frontier).

Although there has been an increasing interest in MOO, there is a lack of Multi-Objective Optimization benchmarks as well as of standards for comparing their performance. Comparing to Single-Objective, Multi-Objective metrics are much harder to define, as they must evaluate a collection of vectors representing a non-dominated set instead of a single scalar value. Whereas we can use univariate statistical tests on single scalar values, other techniques must be applied to compare nondominated collection of vectors, i.e., the Pareto Front.

Multi-Objective Optimization might be addressed differently according to other factors, such as the preference for certain objectives [2]. In this case, the multi-objective problem is reduced to a single-objective problem by means of a preference articulation, being the Weighted Sum (or Linear Scalarization) the most commonly used, which combines the objectives into a single function to be optimized. Therefore, the evaluation of the algorithm's performance is now easier, since we are looking for the single scalar value that optimizes a particular combination of objectives. A different way to approach MOO is to yield a good approximation to the true Pareto front.
**[<TO CONTINUE]**




###### References

 [1] Khazaii, J. (2016). Advanced Decision Making for HVAC Engineers: Creating Energy Efficient Smart Buildings. Springer

 [2] Knowles, J., & Corne, D. (2002). On Metrics for Comparing Nondominated Sets. In Proceedings of the 2002 Congress on Evolutionary Computation, CEC 2002 (pp. 711–716).


#### Initial Overview over the Julia dependencies (last updated October, 2nd, 2019):

In an initial stage, we opt not to explore these packages as we are focusing in a Multi-Objective context. These packages provide statistical or information-theory related metrics which are not directly applicable to the assessment of the performance of different Multi-Objective Optimization Algorithms (MOOAs).

  |  Applicable  |  Library   |
  |:------------:|:---------- |
  | Maybe        | [InformationMeasures](https://github.com/Tchanders/InformationMeasures.jl): Entropy, mutual information and higher order measures from information theory, with various estimators and discretisation methods. |
  | Maybe        | [Shannon](https://github.com/kzahedi/Shannon.jl): Entropy, Mutual Information, KL-Divergence related to Shannon's information theory and functions to binarize data. |
  | Maybe        | [Jackknife](https://github.com/ararslan/Jackknife.jl): Jackknife resampling and estimation in Julia. |
  | Maybe        | [LARS](https://github.com/simonster/LARS.jl): Least angle regression and the lasso covariance test. |
  | Maybe        | [MIPVerify](https://github.com/vtjeng/MIPVerify.jl):Evaluating Robustness of Neural Networks with Mixed Integer Programming. |
  |~~No~~ | ~~[MCMCDiagnostics](https://github.com/tpapp/MCMCDiagnostics.jl): Markov Chain Monte Carlo convergence diagnostics in Julia.~~ |


- The measures provided by the first two packages (InformationMeasures and Shannon) are similar and founded on information-theory.
- Jackknife package might be interesting to automatically generate different runs and gather statistically relevant measures.
- MIPVerify will only make sense if Neural Networks are implemented.
