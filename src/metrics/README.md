# Metrics

Multi-Objective Optimization problems (MOO) belong to the set of problems
concerned with the optimization of more than one objective simultaneously.
The addition of other, potentially conflicting, objectives to the optimization
process prompts the re-definition of optimality. While in Single-Objective
problems we expect the optimal solution to be the set of parameters that
achieve the best objective value possible, in multi-objective the best possible
configuration for one of the objectives is rarely the best possible
configuration for all other objectives as well. This may be explained by the
fact that the objectives are often contradictory.

In order to be able to compare different multi-objective solutions, these
problems are often addressed considering the Pareto optimality (or Pareto
Efficiency) concept [1]. This concept, named after the economist Vilfredo
Pareto, defines an optimal solution as being a solution for which it is
impossible to improve an objective value without deteriorating others.
Such a solution is also said to be non-dominated or noninferior. The set of
non-dominated solutions is called the Pareto Front (or Pareto Frontier).

Although there has been an increasing interest in MOO, there is a lack of
Multi-Objective Optimization benchmarks as well as of standards for comparing
their performance. Comparing to Single-Objective, Multi-Objective metrics are
much harder to define, as they must evaluate a collection of vectors
representing a non-dominated set instead of a single scalar value. Whereas we
can use univariate statistical tests on single scalar values, other techniques
must be applied to compare nondominated collection of vectors, i.e., the Pareto
Front.

Multi-Objective Optimization might be addressed differently according to other
factors, such as the preference for certain objectives [2]. In this case, the
multi-objective problem is reduced to a single-objective problem by means of a
preference articulation, being the Weighted Sum (or Linear Scalarization) the
most commonly used. The Weighted Sum preference articulation combines the
objectives into a single function to be optimized. Therefore, the evaluation of
the algorithm's performance is now easier, since we are looking for the single
scalar value that optimizes a particular combination of objectives. A different
way to approach MOO is to yield a good approximation to the true Pareto front.


**[<TO CONTINUE]**
/
Ideas to elaborate:
 - Compare MOOAs involves comparing the nondominated sets returned by each
 algorithm [2].
 - In general it is hard to select a measure to compare these sets [2] (e.g. if T
   is the true Pareto front, how do we compare a algorithm which produces a
   result with a single point a that belongs to T,  with an algorithm that
   produces a widespread set of nondominated points either dominated by a or in T)
 - It is stated by [3] that displaying the pareto fronts in graphical forms
   allows to better understand which algorithm performs best.
 - Non-Dominated Set Comparison Metrics (NDSCM)
 - In [4] they identify three goals to compare and contrast NDSCM (they are
   however fragile goals [2]):
      1) Distance to the True Pareto Front should be minimized.
      2) Solutions should be well (in most cases uniformly) distributed in the
      objective space.
      3) Extent of the obtained nondominated front should be maximized, i.e.,
      for each objective, a wide range of values should be present.
  - [5] Considered the problem of evaluating approximations to the true pareto
    front. They define outperformance relations that express the relationships
    between two sets of internally nondominated objective vectors (weak, strong,
    complete outperformance).
  - [2] Metrics may be considered acording to if they induce total ordering or
  according to its cardinality.
/

When evaluating the performance of the result of two MOOAs, one can use a
direct comparison, a reference, or an independent metric[2]. In the first metric
type, the results of one algorithm are scored directly against the results of
the other, whereas in the reference metric, the scoring is first done against a
reference set, usually representing the true Pareto front, and only then are those
two scores compared. In an independent metric, each algorithm is scored according
to properties that do not depend on any other set.



###### References

 [1] Khazaii, J. (2016). Advanced Decision Making for HVAC Engineers: Creating
 Energy Efficient Smart Buildings. Springer

 [2] Knowles, J., and Corne, D. (2002). On Metrics for Comparing Nondominated
 Sets. In Proceedings of the 2002 Congress on Evolutionary Computation,
 CEC 2002 (pp. 711–716).

 [3] Veldhuizen, D. V. (1999). Multi Objective evolutionary algorithms:
 Classifications, Analysis, New Innovations. Multi Objective evolutionary
 algorithms. Air Force Institute of Technology, Wright Patterson, Ohio.

 [4] Zitzler, E., Deb, K., and Thiele, L. (2000). Comparison of multiobjective
 evolutionary algorithms: empirical results. Evolutionary Computation, 8(2),
 173–195.

 [5] Hansen, M. P., and Jaszkiewicz, A. (1998). Evaluating the quality of
 approximations to the non-dominated set. IMM Technical Report IMM-REP-1998-7,
 31.

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
