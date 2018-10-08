# Multi-Objective Optimization Frameworks

In order to fulfill with the goal of providing a broader set of optimization algorithms, we reviewed the most reputed Multi-Objective Optimization (MOO) open-source libraries. After a general overview, we narrowed them to six fundamental libraries that provide wider sets of Multi Objective Optimization Algorithms (MOOAs), while also providing a few Single Objective Algorithms. Other key points for the selection of these libraries were their ease of integration, the reputation of the algorithms, i.e., if they are actively used and widely known by optimization practitioners, and whether they are being actively maintained. Additionally, to further narrow down the number of optimization libraries to integrate with, a more extensive research was done. The following sections present the results for the six studied libraries: Platypus, PyGMO, and OTL/PyOptimization in Python, MOEA Framework and jMetal, in Java, and Paradiseo in C++.

Although the developed framework is being developed in Julia, at the time of the research no relevant MOO frameworks were provided.

| Language  | Frameworks |
|:---------:|:----------:|
| Python    | Platypus, PyGMO, PyOTL, DEAP, Inspyred, PyOMO, Sci-py, skopt, NLopt, pySOT, RBFOpt, Py.earth, CNVXOpt |
| Java      | ECJ, Opt4J, jMetal, MOEA Framework |
| C++       | Paradiseo  |


## [Platypus](https://platypus.readthedocs.io/en/latest/)

Platypus, implemented in Python, is presented as an open-source libary focused on Multi-Objective Evolutionary Algorithms (MOEAs). Besides providing a vast number of algorithms, Platypus also provides a set of standard benchmark problems and indicators to assess the performance of the algorithms.

Despite not being extensively documented on the official pages, Platypus' API is rather logical and intuitive, hence facilitating its usage. Additionally, further information regarding the provided data structures of the API might be found next to its implementation on the [Github repository](https://github.com/Project-Platypus/Platypus). Its implementation is clear and understandable, modular, easily extensible, and with a single dependency.


From the architecture perspective, Platypus defines five core components that together compose the Optimization concept: (1) Problem, (2) Algorithm, (3) Solution, (4) Constraint, (5) Type, (6) TerminationCondition, and (7) Dominance. The Problem component represents an optimization problem, thereby storing information about the objective functions, the constraints if any, the types of decision variables as well as their bounds. The Algorithm component is an abstraction for the definition of the overall protocols for evaluating Solutions and executing runs of particular algorithms. The Algorithm makes use of the TerminationCondition component to verify when the run should be terminated. Solutions are defined by the particular problem they belong to, the decision variables and the obtained objective values, as well as the information regarding its feasibility and violation of constraints. Moreover, Platypus provides the possibility not only to perform both equality and inequality constrained problems by providing the Constraint component, but also to differentiate decision variables in continuous and discrete variables. The Type component therefore encloses both the nature of each decision variable, but also its bounds. Moreover, since this is a MOO library the Dominance component is also a crucial for establishing the comparison of two multi-objective solutions.

Platypus provides two Single Objective Algorithms: the Genetic Algorithm (GA) and the Evolution Strategy (ES), and 12 Multi-Objective Algorithms: CMAES, Epsilon-MOEA, GDE3, IBEA, MOEA/D, NSGA-II, NSGA-III, PAES, PESA2, SPEA2, OMOPSO, and SMPSO.

Regarding the performance indicators, it seems to support the most widely used indicators, such as the Generational Distance (GD), the Inverted Generational Distance (IGD), the Epsilon indicator, the Spacing, and the Hypervolume (HV).

Platypus also provides a module called Experimenter that allows to easily compare the performance of multiple algorithms with different parameters in various test problems. In general, this module, which supports the parallelization of different runs, returns the results of every algorithm run for every problem that was specified.

This framework is available both for Python 2 and Python 3 and has been developed since 2015 and its first release dates April 2018, which means that it is a very recent library. Regarding the maintenance of the library, the most recent commit dates from three months ago, which seems to suggest that the library is still being maintained. Moreover, a testing environment is setup, the tests in the repository do not seem to cover the whole library's functionality, therefore being a drawback of Platypus.



# Conclusions

| Properties    | Platypus         |
|:------------- |:----------------:|
| Architecture  | Well defined     |
| Documentation | Poor             |
| Integration   | Easy             |
| Maintenance   | Seasonal         |
| Tests         | Poor             |
| Popularity/Use| Unknown          |
|


| Algorithm    | Platypus           |
|:------------ |:------------------:|
| SGA          | :heavy_check_mark: |
| ES           | :heavy_check_mark: |
| CMAES        | :heavy_check_mark: |
| Epsilon-MOEA | :heavy_check_mark: |
| GDE3         | :heavy_check_mark: |
| IBEA         | :heavy_check_mark: |
| MOEA/D       | :heavy_check_mark: |
| NSGA-II      | :heavy_check_mark: |
| NSGA-III     | :heavy_check_mark: |
| PAES         | :heavy_check_mark: |
| PESA2        | :heavy_check_mark: |
| SPEA2        | :heavy_check_mark: |
| OMOPSO       | :heavy_check_mark: |
| SMPSO        | :heavy_check_mark: |
| VEGA         | :x:                |
