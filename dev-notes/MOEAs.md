# Evolutionary Algorithms

Evolutionary Algorithms have been largely studied in the field of Multi-
-Objective Optimization. In simulation based optimization, EAs are
particularly important.

In 1999, Zitzler [1] states the lack of comparative studies in the literature,
that provide answers to the following questions:
1 - whether some techniques are in general superior to others.
2 - which algorithms are suited to which kind of problem
3 - what the specific advantages and drawbacks of certain methods


Since 1970s several EA methodologies have been proposed, such as Genetic
Algorithms (GA), Evolutionary Programming (EP), and Evolution strategies (ES).

EA comprises a class of stochastic optimization methods that simulate the
process of natural evolution. The EA approaches operate on a set of candidate
solutions, which are modified by the two principles of evolution: selection and
variation.
  - **Selection** represents the natural mechanism for the survival of the
fittest, i.e., the fittest members in each population are more likely to
reproduce and share their genetic information. EAs use stochastic selection
processes to simulate natural selection in which each solution is given a
chance to reproduce a certain number of times. The number of times they reproduce
depends on their fitness values, representing their quality as individuals.
  - **Variation** represents the natural mechanism by which different traits tend
  to exist within populations. The variation is the result of mutations and
  recombination.
EAs are specially suited for optimization problems due to their flexibility and
adaptability to the task, and to their global search characteristics [1].
Particularly for MOO, EAs allow to capture in the same generation multiple
Pareto-optimal solutions. They have already been identified as being better to
address MOO problems (MOOPs) than blind (or uninformed) search strategies [1].

[TODO-READ more papers. Make a comparison between the statement of Zitzler and
the current state-of-the art algorithms.]

In 1984, Schaffer [2] proposed a vector based EA called Vector-Evaluated
Genetic Algorithm (VEGA). Even though, nowadays VEGA is known for its
limitations, it was one of the pioneering studies concerning MOEAs and inspired
others to develop other MOEAs.

In the following years, several algorithms and
variants of such algorithms were proposed, such as HLGA, FFGA, NPGA, NSGA, and
SPEA. Throughout the years, some of these algorithms have been improved and
newer and more efficient versions have been proposed (e.g. NSGA-II, NSGA-III, SPEA2)

MOEAs provide a gamut of topics to research: niching convergence to
Pareto-optimal Front, elitism, new evolutionary techniques.

Back in 1999, Zitzler claimed there were few comparative studies and that the
ones that existed were restricted to a small subset of algorithms [1]. Today,
almost 20 years after, we still claim that there are few comparative studies,
especially involving MOOAs with different techniques. [TODO - WE NEED
COMPARATIVE STUDIES TO COMAPRE PERFORMANCE OF SURROGATE ASSISTED ALGORITHMS
WITH MOEAs and other TECHNIQUEs!]. Zitzler identified the difficulties in
defining quality measures for MOOAs as one of the main reasons for the lack of
studies in that area.

[TODO  
> "Up to now, there has been no sufficient, commonly
accepted definition of quantitative performance metrics for multiobjective
optimizers"
> There is no accepted set of well-defined test problems in the community.
This makes it difficult to evaluate new algorithms in comparison with existing ones

>The optimization goal for a MOP is based on three objectives:
- distance of the resulting non dominated front to the Pareto Optimal front
should be minimized
- A good (in most cases uniform) distribution of the solutions found is desirable.
- The spread of the non dominated front should be maximized, for each objective
a wide range of values should be covered by the nondominated solutions
]

Generally, an EA is characterized by three facts:
1. set of solution candidates is maintained
2. that set undergoes a selection process
3. is manipulated by genetic operators (recombination and mutation)

Solution candidates are called individuals. The set of solution candidates is called population.
Each individual represents a possible solution (decision vector) to the problem at hand.
The set of all possible vectors constitutes the individual space I.
Population is a set of vectors i ∈ I.

The selection process removes low-quality individuals from the population, while
allowing high-quality individuals to reproduce. This process focus on the search
on promising regions of the search space and to increase the average quality
within the population. This process might be stochastic or fully deterministic.

The quality is represented by a scalar value - the fitness-, which is related
to the objective functions and the constraints. The individual must be decoded
before its fitness is calculated.

Recombination and mutation are responsible for generating new solutions within
the search space by the variation of the existing ones.  There are several
variation operators, which are associated with probabilities to mimic the
stochastic nature of evolution:
 -  Crossover operator: takes a certain number of parents and creates a certain
 number of children by recombining the parents.
 -  Mutation operator: modifies individuals by changing small parts in the
 associated vectors according to a given mutation rate.
Both crossover and mutation operators work in the individual space, and not on
the decoded decision vectors.

The natural evolution is simulated by an iterative computation process. An
initial population is created and a loop simulating evolution is then ran for a
certain number of times. This loop consists of the selection and variation
operators, discussed previously. Each loop iteration is called a generation. At
the end, the outcome of the EA is the best individuals found during the entire
evolution process.

```
Input:  N   (Population Size)
        T   (maximum number of generations)
        pc  (crossover probability)
        pm  (mutation rate)
Output: A   (nondominated set)  

Step1.  Initialization: Set P0 = {} and t = 0.
        For i = 1, ..., N do
          Choose i ∈ I according to some probablity distribution
          Set P0 += {i}

Step2.  Fitness Assignment
        For each individual i ∈ P do
          Determine the encoded decision vector x = m(i)
          Determine the objective vector y = f(x)
          Compute the scalar fitness value F(i)

Step3.  Selection: Set P' = {}.
        For i = 1, ..., N do
          Select one individual i ∈ P' according to a scheme and its
          fitness value F(i)
          Set P' += {i}
          The temporary population P' is called the mating pool.

Step4.  Recombination: Set P'' = {}.
        For i = 1, ..., N/2 do
          Choose two individuals i,j ∈ P' and remove them from P'
          Recombine i and j. The resulting children are k, l ∈ I
          Add k, l to P'' with probability pc. Otherwise add i,j to P''.

Step5.  Mutation: Set P''' = {}.
        For each individual i ∈ P'' do
          Mutate i with mutation rate pm. The resulting individual is j ∈ I
          Set P''' += {j}

Step6.  Termination: Set Pt+1 = P''' and t=t+1
        If t >= T or other stopping criterion
          A = p(m(Pt))
        else
          Go to Step2.       

```

#### Fitness Assignment and Selection
  1. **Selection by Switching Objectives**: Decides which member of the population
  goes into the mating pool by assessing the quality of a different objective.

  2. **Aggregation Selection w/ Parameter Variation**: The objectives are aggregated
  into a single parameterized objective function (potentially different for each
  individual)

  3. **Pareto-based Selection**: Uses a ranking procedure based on Pareto dominance.
  Temporarily removes all the non-dominated individuals and assign them rank one.
  Then, do the same to the next nondominated individuals, which are assigned rank two.
  Their rank determines their fitness value.

#### Population Diversity
Maintaining a diverse population is crucial for the efficacy of an MOEA. Etilist
approaches tend to converge towards a single solution and often loses solutions
due to three factors:
  - selection pressure, defined in terms of takeover time, i.e., the time
  required until the population is completely filled by the best individual.
  - selection noise, defined by the variance of a selection scheme
  - operator disruption, destructive effects which recombination and mutation
  may have.
To overcome these factors, several methods have been developed:

  1. **Fitness Sharing**: promote the formulation and maintenance of stable
  subpopulations (niches). The shared fitness F(i) of an individual i ∈ is equal
  to its old fitness F'(i) divided by its niche count.

  2. **Restricted Mating**: two individuals are only allowed to mate if they are
  within a certain distance of each other.

  3. **Isolation by distance**: each individual is assigned a location. Each
  subpopulation  evolves separately

  4. **Overspecification**: The individual contains active and inactive parts.
  Information can be hidden during evolution (inactive parts have no function).

  5. **Reinitialization**: Reinitialize the whole parts of the population after
  a certain period of time or whenever the search stagnates.

  6. **Crowding**: New individuals (children) replace similar individuals in the
  population. Only a few individuals are considered at a time.


#### Elitism

Policy to always include the best individual of Pt into P(t+1) in order to
prevent losing it due to sampling effects or operator disruption This strategy
which can be extended to copy the best b selections to the next generation is
denoted as elitism. In multi-modal functions it may cause premature convergence.


### Overview of Evolutionary Techniques

1. Schaffer's Vector Evaluated Genetic Algorithm (VEGA)
    - Selection by switching Objectives (selection for each of k objectives
      separately, fill equally sized portions of the mating pool)
    - Mating pool is shuffled and crossover and mutation performed as usual
    - Fitness proportionate selection
    - Serious drawbacks
    - Strong reference point

2. Hajela and Lin's Weighting-based GA (HLGA)
    - Aggregation selection w/ parameter variation
    - Based on the weighting method and to search for multiple solutions in
    parallel
    - Weights are encoded in the individual's vector. Each individual is evaluated
    with regard to a potentially different weight combination.
    - Fitness sharing in Objective space.
    - Weighting method is biased towards convex portions of the trade-off front.
    - Widespread due to its simplicity

3. Fonseca and Fleming's Multi-Objective GA (FFGA)
    - Pareto-based ranking procedure, where an individual's rank equals the
    number of solutions encoded in the population by which its corresponding
    decision vector is dominated.
    - Mating pool is filled using stochastic universal sampling.
    - Adaptive fitness sharing and continuous introduction of random immigrants
    are concepts that were extended after FFGA.

4. Horn, Nafpliotis, and Goldberg's Niched Pareto GA (NPGA)
  - Tournament selection
  - Pareto dominance

5. Srinivas and Deb's Nondominated Sorting Genetic Algorithm (NSGA)
  - Different trade-off fronts in the population are removed and ranked.
  - Fitness sharing is performed for each front separately in order to maintain
  diversity (in the decision space)
  - High fitness values correspond to high reproduction probabilities
  - Stochastic remainder selection.

6. Strength Pareto Evolutionary Algorithm (SPEA)
  - Introduced by Zitzler
  - Stores those individuals externally that represent a nondominated front among
  all solutions considered so far.
  - Pareto dominance to assign scalar fitness values to individuals
  - Clustering to reduce the number of individuals externally stored without
  destroying the characteristics of the trade-off front.
  - THe fitness of a population member is determined only from the individuals
  in the external set.
  - All individuals in the external set participate in selection.
  - Pareto-based niching method is provided in order to preserve diversity in
  the population

# Comparing MOEAs

  - Scale independent indicators: Uses coverage, but instead of C-metric,
  proposes using the D metric.
  - Scale dependent indicators: Uses M-family (M1 - Avg dst to Pareto optimal
    set, M2 - distribution combined w/ the number of nondominated solutions, M3 - spread of A)

# Results
  - Hierarchy of algorithms emerged regarding the distnace to the Pareto-optimal
  front in descending order of merit: SPEA, NSGA, VEGA, NPGA and HLGA, FFGA.

---- end [1] ----
-------------------------------------------------------------------------------

# 


References

1 - Zizler: Evolutionary Algorithms for Multiobjective Optimization: Methods
 and Applications

2 - Schaffer, J. D. (1984). Multiple Objective Optimization with Vector
Evaluated Genetic Algorithms. Ph. D. thesis, Vanderbilt University. Unpublished
