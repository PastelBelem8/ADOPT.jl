MOEAFramework# Multi-Objective Optimization Frameworks

In order to fulfill with the goal of providing a broader set of optimization algorithms, we reviewed the most reputed Multi-Objective Optimization (MOO) open-source libraries. After a general overview, we narrowed them to six fundamental libraries that provide wider sets of Multi Objective Optimization Algorithms (MOOAs), while also providing a few Single Objective Algorithms. Other key points for the selection of these libraries were their ease of integration, the reputation of the algorithms, i.e., if they are actively used and widely known by optimization practitioners, and whether they are being actively maintained. Additionally, to further narrow down the number of optimization libraries to integrate with, a more extensive research was done. The following sections present the results for the six studied libraries: Platypus, and OTL/PyOptimization in Python, MOEAFramework and jMetal, in Java, and, Paradiseo and PaGMO in C++.

Although the developed framework is being developed in Julia, at the time of the research no relevant MOO frameworks were provided.

| Language  | Frameworks |
|:---------:|:----------:|
| Python    | Platypus, PyOTL, DEAP, Inspyred, PyOMO, Sci-py, skopt, NLopt, pySOT, RBFOpt, Py.earth, CNVXOpt |
| Java      | ECJ, Opt4J, jMetal, MOEAFramework |
| C++       | Paradiseo, PaGMO  |


## [Platypus](https://platypus.readthedocs.io/en/latest/)

Platypus, implemented in Python, is presented as an open-source library focused on Multi-Objective Evolutionary Algorithms (MOEAs). Besides providing a vast number of algorithms, Platypus also provides a set of standard benchmark problems and indicators to assess the performance of the algorithms.

Despite not being extensively documented on the official pages, Platypus' API is rather logical and intuitive, hence facilitating its usage. Additionally, further information regarding the provided data structures of the API might be found next to its implementation on the [Github repository](https://github.com/Project-Platypus/Platypus). Its implementation is clear and understandable, modular, easily extensible, and with a single dependency.

From the architecture perspective, Platypus defines five core components that together compose the Optimization concept: (1) Problem, (2) Algorithm, (3) Solution, (4) Constraint, (5) Type, (6) TerminationCondition, and (7) Dominance. The Problem component represents an optimization problem, thereby storing information about the objective functions, the constraints if any, the types of decision variables as well as their bounds. The Algorithm component is an abstraction for the definition of the overall protocols for evaluating Solutions and executing runs of particular algorithms. The Algorithm makes use of the TerminationCondition component to verify when the run should be terminated. Solutions are defined by the particular problem they belong to, the decision variables and the obtained objective values, as well as the information regarding its feasibility and violation of constraints. Moreover, Platypus provides the possibility not only to perform both equality and inequality constrained problems by providing the Constraint component, but also to differentiate decision variables in continuous and discrete variables. The Type component therefore encloses both the nature of each decision variable, but also its bounds. Moreover, since this is a MOO library the Dominance component is also crucial for establishing the comparison of two multi-objective solutions.

Platypus provides two Single-Objective Algorithms: the Genetic Algorithm (GA) and the Evolution Strategy (ES), and twelve Multi-Objective Algorithms: CMAES, Epsilon-MOEA, GDE3, IBEA, MOEA/D, NSGA-II, NSGA-III, PAES, PESA2, SPEA2, OMOPSO, and SMPSO.

Regarding the performance indicators, it seems to support the most widely used indicators, such as the Generational Distance (GD), the Inverted Generational Distance (IGD), the Epsilon indicator, the Spacing, and the Hypervolume (HV).

Platypus also provides a module called Experimenter that allows to easily compare the performance of multiple algorithms with different parameters in various test problems. In general, this module, which supports the parallelization of different runs, returns the results of every algorithm run for every problem that was specified.

This framework is available both for Python 2 and Python 3 and has been developed since 2015 and its first release dates April 2018, which means that it is a very recent library. However, the main developer of Platypus, is also the main developer of MOEAFramework, an older MOEAFramework being developed since 2012. Regarding the maintenance of the library, the most recent commit dates from three months ago, which seems to suggest that the library is still being maintained. Moreover, a testing environment is setup, the tests in the repository do not seem to cover the whole library's functionality, therefore being a drawback of Platypus.

## [PaGMO/PyGMO](https://esa.github.io/pagmo2/index.html)

PaGMO, is a scientific library implemented in C++ providing parallel optimization abstractions for the application of different optimization algorithms [1]. PaGMO also provides a Python API called PyGMO.

Besides providing a vast list of biological-based EAs, PaGMO also provides other iterative algorithms, such as simplex, interior-points, and sequential quadratic programming (SQP) algorithms, which are highly efficient in addressing particular types of problems. For instance, SQP solves nonlinearly constrained optimization problems very effectively, whereas nonlinear and linear convex optimization problems are effectively addressed by interior-points algorithms. These algorithms can be easily combined with EAs to enhance the efficiency of optimization runs.

PaGMO exploits a shared memory model and different threads of execution to exploit the multicore architectures of today's computers. PaGMO follows a generalized island-model paradigm in which each island executes the optimization algorithm in a different thread. The operating system (OS) is responsible for asynchronously managing the schedule of the threads, migrating them as needed in order to balance the system's workload [1]. In this parallelization paradigm, the algorithms are automatically parallelized by exploiting the concept of co-existing populations  that exchange individuals in sparsely distributed moments of time. Instead of centralizing the computation of the fitness values for a population in a single CPU core, smaller subpopulations are created and distributed across different CPU cores, thus achieving a speed-up in the execution of the optimization process. For further details about the generalized island-model, we refer the interested user to [1].

Notwithstanding the benefits of parallelization, PaGMO is rooted on the idea that objective functions are ``fast enough to allow for parallel processing involving a large number of solutions'' which is not applicable in the case of  simulations in the architectural field. In fact, when comparing the long computational time of these simulations with the overhead of the optimization algorithm, the latter is almost negligible. Moreover, adding to the fact that recent versions of simulation software already provide parallelization capabilities to speed-up the calculation, architects often run these optimization and simulation processes in a single machine. Consequently, the benefits of the parallelized model become limited and, can potentially slow down the optimization process, since "over threading" will lead to optimization's tasks running against the simulation's tasks for CPU time.

PaGMO provides an enormous list of Single-Objective Algorithms, and five Multi-Objective Algorithms: VEGA, NSGA-II, NSGA-III, PAES, PESA2, SPEA2, OMOPSO, and SMPSO. In what regards the quality indicators for assessing MOOAs' performance, PaGMO provides a single indicator: the HV.

From the mathematical point of view, this library provides methods to address both constrained and unconstrained optimization problems, as well as problems containing continuous and discrete variables. PaGMO also provides global and local algorithms, which is an advantage as it allows to combine both to balance the exploration and exploitation phases of the optimization process.

The oldest release dates back to 2014, and the most recent version was released on August 2018, suggesting an active maintenance of the library. Regarding its popularity, although there are no indications about the number of users, PaGMO is being used in the European Space Agency, and when looking up at the terms "PaGMO" and "PyGMO" on GoogleTrends, we observe a subtle increase in the number of searches since 2016.

In the online [PaGMO Github repository](https://github.com/esa/pagmo2) it is possible to verify that there is a Continuous Integration test-based environment setup, particularly the tests seem to cover about 99% of the library's code.

Regarding the conceptualization of the library, PaGMO is built around six fundamental concepts: (1) Types, (2) Problem, (3) Algorithm, (4) Population, (5) Island, and (6) Archipelago. PaGMO defines two different base types to support continuous and discrete decision variables. To represent an optimization problem, PaGMO requires modelling an initial problem class exposing methods to compute the fitness function and constraints and to obtain the dimensions of the problem, i.e., the bounds. The optimization problem is then constructed using the previously defined class. Similarly, each optimization algorithm is constructed using a previously defined algorithm class that exposing a method to evolve populations. A population represents a set of potential candidate solutions to a given problem. Each individual solution is represented by an unique ID, a decision vector and its fitness value for a particular problem. The island component is an abstraction encapsulating the algorithm and the population that manages the optimization process by asynchronously using algorithms to continually evolve populations. On the other hand, archipelagos are collections of islands that enable multiple parallel optimizations.
  ```Python
>>> from pygmo import *
>>> prob = problem(rosenbrock(dim = 10))
>>> print(prob)
Problem name: Multidimensional Rosenbrock Function
    Global dimension:                       10
    Integer dimension:                      0
    Fitness dimension:                      1
    Number of objectives:                   1
    Equality constraints dimension:         0
    Inequality constraints dimension:       0
    Lower bounds: [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5]
    Upper bounds: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

    Has gradient: true
    User implemented gradient sparsity: false
    Expected gradients: 5
    Has hessians: false
    User implemented hessians sparsity: false

    Fitness evaluations: 0
    Gradient evaluations: 0

    Thread safety: basic

>>> algo = algorithm(bee_colony(gen = 500, limit = 20))
>>> algo.set_verbosity(100)
>>> pop = population(prob, 20)
>>> pop = algo.evolve(pop)
Gen:        Fevals:          Best: Current Best:
 1             40         261363         261363
101           4040        112.237        267.969
201           8040        20.8885        265.122
301          12040        20.6076        20.6076
401          16040         18.252        140.079
>>> uda = algo.extract(bee_colony)
>>> uda.get_log()
[(1, 40, 183727.83934515435, 183727.83934515435), ...
```

Although PaGMO is fully implemented in C++ for performance reasons, we have said previously that PaGMO also provides a Python interface called PyGMO, that allows to code quicker. Even though, from an integration point of view, there are Julia packages providing interfaces to both languages, due to Python's type compatibility with Julia, Python is more likely to be more quickly integrated. In addition, the documentation for PyGMO is more complete than PaGMO's, providing tutorials and step-by-step guides to configure optimization processes differently.

PaGMO'S API is strongly influenced by the evolutionary and population concepts, and differs from the traditional thinking in which the optimization run is controlled by the algorithm itself. Instead, PaGMO delegates that functionality to the island component that is responsible for asynchronously migrating solutions within islands and evolving populations. For that reason, this API is less intuitive and, necessarily, more verbose.

Moreover, PyGMO's optimization algorithms comprises the ones provided by SciPy, NLopt, GSL, SNOPT and Ipopt frameworks. On the one hand, these extra dependencies enlarge the available set of algorithms and enable the application of algorithms of different categories which can, sometimes, be more appropriate to address some types of problems. For example, PyGMO allows to easily apply model-based (e.g., BOBYQA, COBYLA, NEWUOA) and direct search algorithms (e.g. NMS, Subplex, Praxis), which are provided by NLopt. Other than optimization packages, PyGMO also relies on several advanced visualization packages.


## [MOEAFramework](https://github.com/MOEAFramework/MOEAFramework)

Developed since 2009, the MOEAFramework is a Java framework focused on MOEAs and general purpose optimization algorithms. MOEAFramework also provides tools for the assessment of the algorithms' performance, including tools for performing statistical tests and for identifying an algorithm's key parameters.  

The MOEAFramework focus on providing an extensible API that allows the creation of new algorithms, operators, and problems. This This framework allows their parallel execution in multiple cores.   

To evaluate the performance of the algorithms, MOEAFramework provides quality indicators with different qualities allowing to measure different properties of the Pareto Fronts. These metrics are HV, GD, IGD, MPFE, R-family and the ϵ-indicator.

This framework is very rich in terms of MOEAs, providing thirty-three multi-objective and four single-objective algorithms. MOEAFramework integrates in a single framework the implementations provided by the [Platform and Programming Language Independent Interface for Search Algorithms (PISA)](https://sop.tik.ee.ethz.ch/pisa/?page=pisa.php), by [jMetal](http://jmetal.sourceforge.net/index.html) - another Java framework.

Despite not being documented on the official pages, MOEAFramework's API is elegant and logical. However, the lack of tutorials and information about the dependencies and architecture of the framework, make it more difficult to extend and use this API. Moreover, the available documentation is not up-to-date and the information is not easily extracted from the existing documentation. Notwithstanding its modularity, MOEAFramework severely lacks documentation and application guides, since the guides available are not free.

```Java
NondominatedPopulation result = new Executor()
     .withAlgorithm("NSGAII")
     .withProblem("DTLZ2_2")
     .withMaxEvaluations(10000)
     .withProperty("populationSize", 100)
     .withProperty("sbx.rate", 1.0)
     .withProperty("sbx.distributionIndex", 15.0)
     .withProperty("pm.rate", 0.05)
     .withProperty("pm.distributionIndex", 20.0)
     .run();
```

Regarding the conceptualization of the framework, it is organized in main components: (1) Executor, (2) Problem, (3) Algorithm, (4) Population, (5) Solution. The executor component is the starting point for an optimization process, being responsible for the construction and execution of MOEAs to solve optimization problems. Similarly to the other frameworks, the Problem component exposes two methods, one for evaluating the fitness values and the constraints and other for obtaining new solution to the specified problem. The evaluation methods is then called by each algorithm component, when searching for the optimal solutions in a population. The population is the component representing the set of individual solutions, each of which maintains its unique identifier, decision variables, objectives, constraints, and its attributes. Interestingly, in order to maintain a dynamic feedback of the evolution of the optimization process, MOEAFramework uses an event-based approach to update the information about an optimization run.  

Although there are no reports about the code coverage, after a slight overview of the tests available in the repository, this framework seems to be well tested.

From the integration perspective, the integration of this Java framework with a Julia component is likely to be difficult. Not only the documentation lacks details and tutorials to introduce new users to the framework, but there are also major caveats regarding the functionalities of the interfacing packages provided in Julia. In fact, JavaCall is a Julia package that provides means to call Java programs from Julia code. However, JavaCall has been reported to have some caveats regarding the support of multidimensional arrays and the management of the memory system, which suggest that besides the possibility of having memory leaks, we would also have to deal with type incompatibility issues.


# Conclusions

| Properties    | Platypus          | PaGMO/PyGMO | MOEAFramework |
|:------------- |:-----------------:|:-----------:|:--------------:|
| API           | ✓                | ✖           | ✓              |
| Parallel      | ✖                | ✓           | ✓              |
| Modular       | ✓                | ✓           | ✓              |
| Documentation | ✖                | ✓           | ✖              |
| Integration   | ✓                | ✓           | ✖              |
| Maintenance   | ✓ (Seasonal)     | ✓           | ✓              |
| Tests         | ✖                | ✓           | ✓              |
| Popularity/Use| ⍰ (Unknown)      | ✓           | ⍰ (Unknown)   |
| Discrete/Continuous | ✓          | ✓           | ✓              |
| Constrained/Unconstrained | ✓    | ✓           | ✓              |
| Model-Based Methods  |            | Local - S   | M              |
| Direct-Search Methods|            | Local - S   |               |
| Minimization  | Max/Min           | Min          | Min           |
| Events        |                   |              | ✓             |


### Multi-Objective Algorithms (MOOAs)
| Algorithm    | Platypus | PaGMO/PyGMO | MOEAFramework |
|:------------ |:--------:|:-----------:|:--------------:|
| ES           | ✓ |   |   |
| CMA-ES       | ✓ | ✓ | ✓ |
| ϵ-MOEA       | ✓ |   | ✓ |
| GDE3         | ✓ |   | ✓ |
| IBEA         | ✓ |   | ✓ |
| MOEA/D       | ✓ | ✓ | ✓ |
| NSGA-II      | ✓ | ✓ | ✓ |
| NSGA-III     | ✓ |   | ✓ |
| PAES         | ✓ |   | ✓ |
| PESA2        | ✓ |   | ✓ |
| SPEA2        | ✓ | ✓ | ✓ |
| OMOPSO       | ✓ |   | ✓ |
| SMPSO        | ✓ |   | ✓ |
| NSPSO        |   | ✓ |   |
| VEGA         |   | ✓ | ✓ |
| PADE         |   | ✓ |   |
| HS           |   | ✓ |   |
| AbYSS        |   |    | ✓ |
| Borg MOEA    |   |    | ✓ |
| CellDe       |   |    | ✓ |
| DBEA         |   |    | ✓ |
| DENSEA       |   |    | ✓ |
| ϵCEA         |   |    | ✓ |
| ϵ-NSGA-II    |   |    | ✓ |
| FastPGA      |   |    | ✓ |
| FEMO         |   |    | ✓ |
| HypE         |   |    | ✓ |
| MOCell       |   |    | ✓ |
| MOCHC        |   |    | ✓ |
| MSOPS        |   |    | ✓ |
| Random Search|   |    | ✓ |
| RVEA         |   |    | ✓ |
| SEMO2        |   |    | ✓ |
| SHV          |   |    | ✓ |
| SIBEA        |   |    | ✓ |
| SMS-EMOA     |   |    | ✓ |
| SPAM         |   |    | ✓ |


### Single-Objective Algorithms (SOAs)
| Algorithm    | Platypus           | PaGMO/PyGMO  | MOEAFramework :|
|:------------ |:------------------:|:------:|:----------------:|
| SGA          | ✓                   | ✓     | ✓ |
| DE           |                     | ✓     | ✓ |
| Self-Adaptive pDE |                | ✓     |   |
| Self-Adaptive jDE |                | ✓     |   |
| Self-Adaptive iDE |                | ✓     |   |
| HS           |                     | ✓     |   |
| PSO          |                     | ✓     |   |
| GPSO         |                     | ✓     |   |
| (N+1) ES SEA |                     | ✓     |   |
| SA           |                     | ✓     |   |
| ABC          |                     | ✓     |   |
| xNES         |                     | ✓     |   |
| CS           |                     | ✓     |   |
| COBYLA       |                     | ✓     |   |
| BOBYQA       |                     | ✓     |   |
| NEWUOA       |                     | ✓     |   |
| PRAXIS       |                     | ✓     |   |
| NMS          |                     | ✓     |   |
| SUBPLEX      |                     | ✓     |   |
| MMA          |                     | ✓     |   |
| SLSQP        |                     | ✓     |   |
| CCSA         |                     | ✓     |   |
| LS-BFGS      |                     | ✓     |   |
| Preconditioned truncated Newton  | | ✓     |   |
| Shifted limited memory var-metric| | ✓     |   |
| Ipopt ?      |                     | ✓     |   |
| SNOPT ?      |                     | ✓     |   |
| WORHP ?      |                     | ✓     |   |
| ES           |                     |       | ✓ |
| RSO          |                     |       | ✓ |



### Metrics

| Indicator | Platypus | PaGMO/PyGMO | MOEAFramework |
|:--------- |:--------:|:-----:|:------:|
| GD        | ✓       |        | ✓ |
| IGD       | ✓       |        | ✓ |
| HV        | ✓       | ✓      | ✓ |
| ϵ-Indicator | ✓     |        | ✓ |
| MPFE      |          |        | ✓ |
| R-Family  |          |        | ✓ |
| Spacing   |          |        | ✓ |





Reference:

[1] Ignacio Hidalgo, J., Fernandez, F., Lanchares, J., Cantú-Paz, E., and Zomaya, A. (2010). Parallel architectures and bioinspired algorithms. Parallel Computing, 36(10–11), 553–554.
