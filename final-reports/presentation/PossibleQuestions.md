# Possible Questions

This file encloses possible questions that the panel might pose during the presentation. 
In addition to the questions, here lie possible answers to each question.


Moreover, I also leave here some of the papers that I should review prior to the discussion, 
as I barely remember what they state. 

- [ ] Waibel 2019
- [ ] Custodio 2011
- [ ] Custodio 2018
- [X] Hooke-Jeeves 1961 
	Hooke Jeeves is a pattern search algorithm that, given an initial solution, searchs in every dimension the direction of best improvement and then keeps moving in that direction until no improvement is seen. Afterwards, it repeats the process by exploring every dimension again.
- [ ] Nelder Mead Simplex (review the geometric operations - shrinking, contraction, expansion, reflection)
- [ ] Thomas Rowan 1989 - Subplex
- [ ] OMOPSO 2005 
- [ ] SMPSO 2009
- [ ] GDE3 
- [ ] HyPE
- [ ] IBEA
- [X] SPEA2
- [x] Wortmann's paper comparing OPOSSUM to MOOAs (review how he assesses its performance)
- [X] Hussein and Deb 2016, 
- [X] Zapotecas-Martinez 2016, 
- [X] Diaz-Manriquez 2016

-------------------------------------------------------------------

## A. Complexity
1. Throughout all your documents you describe the time complexity as a major problem of these problems, even proposing to identify the most adequate algorithms that allow to reduce the time necessary to find good solutions. However, you never mention the space complexity. Is this a problem? Have you consider it? 

	A: This is a complicated subject to address. In terms of memory, if we analyse them by category, we have that: 
	- __direct-search__: most of the studied algorithms either require __n_dims__+1 solutions, where __n_dims__ is the number of variables of the problem. Both simplex and directional direct-search methods store in general __n_dims__+1 solutions. 
	Recently, a paper that explores the application of DIRECT to MOO (MO-DIRECT) suggests an alternative to MOEAs which requires O(__n_evals__), which might be problematic when submitted to narrow space constraints.

	- __metaheuristics__: most of the algorithms that we apply in this dissertation either store N solutions (where N is the number of individuals of a population) or they merely store 1 solution (e.g., Simulated Annealing, PAES, hill climbing).

	- __model-based__: are more complicated to measure. In general, they store all the solutions, to guarantee a good coverage of the solution space and a good approximation. However, one can think of an adaptive strategy which becomes more discerning with time, forgetting the solutions further way from the direction of the search or using clustering techniques to obtain a representative point of certain regions of the solution space.

	- __sampling algorithms__: the most naive implementation would have in memory at the same time *n_sample* solutions. However, one can think of making it constant, if we sequentialize the generation of the samples (e.g., storing the seed for the sampling algorithms and recomputing the bins for instance in the LHS, or by just iteratively randomly sampling the solutions in MC algorithms)

	Overall, I would say that this is becoming less of a concern with increasing complexity of MOPS where the evaluation budgets are usually within the practical storage limits.

2. The main conclusion of your dissertation seems to be __test different algorithms__, however this spends evaluations. Wouldn't it be better to apply those evaluations to run a longer optimization process which could achieve better solutions? Why test several algorithms? 

	A: Focusing those evaluations on a single algorithm can be better if the algorithm turns out to be the most efficient or appropriate for that specific problem. However, in general, we do not know a priori for these problems which algorithm performs best and by settling for a single algorithm, we can be wasting evaluations with a poor algorithm which will never be able to retrieve near optimal solutions. For instance, there's a wide adoption of Genetic Algorithms and, particularly, Evolutionary Algorithms. However, most of these algorithms focus on an initial set of solutions and from there eventually become more focused on specific regions of the solution space (i.e., we observe an intensification of the traits that were identified as the best solutions, i.e., the fittest ones, however, these are biased by the solutions that were initially sampled and due to that they may never reach near optimal solutions). 

3. **You state in your preprint document that the complexity of optimization processes is exponential on the number of objectives**. Why so?
	
	A: I do believe this might be an error. I think that the complexity of optimization processes is exponential on the number of variables as a brute force approach would require us to test all the possible combination of values for each variable, in the worst case. Eventually, the objective space would be exponential as there are also several combinations among the different objectives, but we do not actually manipulate these values freely in an optimization process, we merely describe them in terms of the variables.   

	No caso limite, seria testar para cada objetivo todas as soluções possíveis.



4. In page 21, you state that the experimental approach is less time consuming than the other 3 approaches, but given that this is an uninformed search strategy this seems odd. Explain. 

	A: That statement comes from the fact that typical experimental approaches are mainly guided by a pre-defined number of samples, which I assumed to be smaller than the limit of the other approaches. First and foremost, I believe there are two main points to consider in this case:
		- Quality of the results is not considered for this statement, I consider the time complexity as being merely measured through the total number of evaluations
		- The number of evaluations, which I considered to be smaller than the limit for the other approaches. While this might be strange, this approach can be substantially faster than others if the user has already good knowledge about the problem and can use it to narrow the solution space just enough to require fewer evaluations.  

5. What is the exploitation vs exploration problem?  (page 29)
	
	A: Exploitation consists in searching the most promising regions for the optimal solutions, whereas exploration consists in the search of wider regions of the space (instead of considering only the promising regions), so as to guarantee that there is not another promising region, which is better than all others. The exploitation vs exploration problem basically consists in balancing this trade-off between what we already know of the solution space and what we do not know. By focusing evaluations on the already known promising regions we might be wasting evaluations in spaces which are not the best possible. But focusing on searching other unknown regions, we might be wasting evaluations with regions which might not be good. 
	Exploration should be larger in initial stages of the search, whereas exploitation should be considered in final stages of the search. 


-------------------------------------------------------------------

## B. Algorithms

1. What is combinatorial optimization? 

	A: Is a subset of mathematical optimization related to operations research and algorithm theory. Consists of finding an optimal object from a finite set of objects. These problems often present a set of feasible solutions that is discrete (or that can be reduced to discrete) and which we aim to find the best solution. Example of problems are the travelling salesman problem (TSP) and the minimum spanning tree (MST).

2. Can't we apply the methods from Mixed Integer Optimization to these problems? Why? 

	A: Well, in general we do not know anything about the behavior of these objective functions. And lots of these algorithms make assumptions on the relations between each variable (e.g., they must be linear or quadratic). Moreover, most of these algorithms cannot properly deal with non-convex, multi-modal or non-differentiable optimization problems. Some of those algorithms either explore derivatives or they rely on other methods like ...

	Tinha um âmbito delimitado, delimitamos c/ algoritmos que surgem na literatura como mais apropriados para este tipo de problemas.

3. In page 24, you state that model-based methods are deterministic. However, these are often explored by random strategies. What is your comment on this? 

	A: The algorithms themselves are deterministic, i.e., the surrogates themselves are always built in the same manner. Given the same configurations and the same input data, they will always retrieve the same model. However, the strategies that explore the surrogates can present some randomness (e.g., sampling methods or Metaheuristics). 
	Note however, that these classifications are a bit thin in the sense that depending on the point of view they can be perceived as belonging to other categories. This is the reflection of the lack of standards.

4. In page 27, you describe the SPEA2 algorithm, however it remains unclear what happens if the archive is updated with a solution that dominates the other half of the archive. These will be removed, but will the archive be filled with other solutions? Which ones?

	A:  The process goes like this: If the number of nondominated solutions fit into the archive, then the archive becomes this set of solutions. Otherwise, it means that either the archive is too small or that it is too large. 
	In the first case, after placing the nondominated solutions in the archive, the best dominated individuals in the previous archive and the population are copied to the archive. 
	In the second case, a truncation operator is required. This operator iterates over all solutions and at each iteration it removes the solution which has the minimum distance to another individual is chosen at each stage (if there is a tie, consider also the second minimum distance and so on...).

5. You mention more than once that metaheuristics lack convergence guarantees. Nevertheless it is possible nowadays to find papers proving their convergence. What do you have to say about that?

	A: [TODO-LER] Convergence Analysis of Metaheuristics, Gutjahr 2009
	Despite the existence of convergence proofs for some of the metaheuristics algorithms, I have the feeling that in general they make strict assumptions about the conditions under which these proofs are ensured, conditions which do not verify in the types of problems involved in our work. They typically require several hundreds or thousands of evaluations and sometimes other guarantees on the parameters.

6. Why did you opt for these algorithms instead of selecting others (e.g., Why haven't you evaluated the HJ algorithm) ? 

	A: In terms of the SOO algorithms, these seem to be widely known in the literature and have already been made available in ready to use tools. Moreover, for the particular case of direct-search we opted for adopting NMS and Subplex instead of HJ algorithms and other well-known algorithms, because with multidimensional problems, a set of vertices provides much more information than a single solution (which is normally what is required by other direct search methods). 
	I know there have been some studies involving other algorithms (e.g., HJ+PSO). Currently the framework does not support such algorithms, but it is extremely easy to add one such implementation. 

	Não excluo e noutros trabalhos poderiam ser explorados outros algoritmos.

7. How does each algorithm work?
	
	A:	MONSS (Zapotecas-Martinez and Coello Coello, 2016) - Algorithm for box-constrained MOO problems. Adopts a nonlinear simplex search scheme in order to obtain multiple elements of the Pareto optimal set. The search is directed by a well-distributed set of weight vectors, each of which defines a scalarization problem (using the Tchebycheff approach) that is solved by deforming a simplex according to the movements described by NMS method.

8. Describe other global model-based strategies (e.g., RF, MLP, GPR).

	A:

9. Describe the HypE algorithm?

	A: 

-------------------------------------------------------------------

## C. Indicators

1. HV seems to be the most accurate performance indicator, why do you use the other indicators to evaluate the algorithms' performance?

	A: HV is accurate but it has some limitations regarding its time complexity depending on the number of (objective) dimensions, the number of points or solutions resulting from the optimization and whether we need an exact result or not. In general, it is used to evaluate the solutions during the execution of MOOAs but it can be used to measure the MOOA's quality. 

	Although the evaluated case studies do not suffer from the aforementioned limitations, as they typically present a lower number of objectives (<7), number of solutions inferior to a few hundreds at most (<500), we aimed at performing a more extensive analysis, also to have a clearer picture of how the other metrics could relate and translate in good or bad values.

2. You describe several performance indicators in your work. Are there others? 
	
	A: Yes, there are. As unary indicators, we have the diversity measure, the entropy, the Deb's spacing among others. 
	As for binary indicators, there is the two set coverage.
		- __Entropy__ uses Shannon's entropy concept to measure the uniformity of the approximation set distribution. This indicator makes the assumption that each solution provides some information about its vicinities, thus modeling each solution with Gaussian distributions. These distributions add up to form a density function capable of identifying peaks and valleys in the objective space, corresponding to dense and sparse zones, respectively. 
		
		- __Diversity Metric__ is similar to the Entropy indicator. However, it projects the solutions of both the approximation set and the reference set to a hyperplane which is subdivided uniformly. It assigns each interval two numbers: one number marking whether that interval contains at least one optimal solution in the reference set, and the second number marking whether the interval in addition to the optimal solution in the reference set, also contained at least one solution in the approximation set. Then, the diversity measure is the sum of the score of each interval, which are assigned using a sliding window technique (considering one interval and its immediate neighbors) based on the value of the marks\footnote{The scoring function considers the distribution of the marks in three consecutive grids.}. So, the diversity of the reference set considers the value of the first marks, whereas the diversity of the approximation set considers the values of the second marks. In the end, the diversity metric is given by the relative difference between the diversity of the approximation set and the diversity of the reference set. The best diversity possible is achieved if all intervals enclose at least one point\cite{Deb2002DM}. Then, it computes the ratio between the number of intervals that have at least one nondominated solution from both sets and the number of intervals that have at least one nondominated solution of the reference set. Higher values of the diversity metric imply a better distribution and higher diversity of the approximation set when compared to the reference set itself. 

		- __Two Set Coverage__: yields the number of solutions in one approximation set that are dominated by at least one of the solutions of the other approximation set. A value of 1 suggests that the second approximation set is completely dominated by some solutions in the first one, whereas a value of 0 represents the situation when none of the solutions of the second approximation set is covered by the first approximation set.)

3. What is the main difference from GD and IGD? 

	A: GD measures the approximation of the aPF to the tPF, whereas the IGD measures the approximation of the tPF to the aPF. aPF is in a certain why biased as it does not ensure that the aPF is diverse. If a front is within the same region of the solution space the GD measure will be very good, which means that the PF is converging/close to the tPF. However, this is not a really good solution. IGD is better in that sense because it uses every point in the tPF to measure how good the aPF is.

[TODO] 4. What algorithm did you used for HV? What are its main limitations? Are those applicable to the evaluated case studies?

	A: I used the Quick Hypervolume implementation of Alexandre Francisco and Luis Russo. If I understand correctly they ideas similar to the quicksort to reduce the computational effort of the hypervolume computation. There were also other algorithms which can incurr in smaller computational efforts for the problems we face in architecture such as HSO (Hypervolume by Slicing Objectives (HSO)), I have also seen some algorithms that return approximations to the HV indicator, but the complexity does not justify. That would be more applicable if we were to use hypervolume within any algorithm (such as HypE).

-------------------------------------------------------------------

## D. Evaluation 

1. There is a difference between the evaluation your performed in your SOO case study and the one in MOO, namely, you do not draw conclusions about the time each algorithm takes and whether they are optimal. How could you solve it? 

	A: Well, the problem here is that with the increase in the complexity and the shift from scalar-valued solutions to vector-valued solutions we stop having a straightforward way of measuring the quality of the solutions and we have multiple optimal solutions, being the optimality measured in terms of the Pareto Optimal concept. Moreover, the lack of standards regarding the best way to measure the quality of each algorithm makes this a more difficult task.

		If the true Pareto front is known we could measure it with the number of optimal solutions achieved by each algorithms in terms of the number of evaluations. However, in architecture this is not usually known and it is impractical to obtain (given the large computational effort required).

		In architecture, a recent study (Wortmann 2017 - OPOSSUM) has compared the performance of SOOAs with the HypE MOOA, and has measured this performance evolution in terms of the speed of convergence and the stability of each algorithm. 
			- Speed of convergence refers to how fast algorithms approach the optimum, measured as the improvement per function evaluation
			- Stability refers to the reliability of optimization algorithms and is a concern especially for stochastic algorithms such as metaheuristics (measured by repeatedly running an optimization algorithm on the same problem).

			- To compare the solutions retrieved by HypE, the author calculates the SOO value for the solutions found by HypE with an weighted objective function. But they store for each SOOA and MOOA the individual values of the objectives to plot the Pareto Front of each algorithm.

		Although this approach allows a mean to compare SOO and MOOA it does not allow to clearly compare several MOOAs. To that end, one could use the HV indicator and monitor for each evaluation how this indicator improves overtime, thus allowing to understand the evolution of the PFs and how diverse the solutions are.  

2. There are already some studies in engineering and science that evaluated MOOAs in the context of MOOs. How did they evaluate the different algorithms? How were these compared? (e.g., Hussein and Deb 2016, Zapotecas-Martinez 2016, Diaz-Manriquez 2016)

	A: In Hussein and Deb they compare it using a combination of the IGD indicator and Pareto Front plots. They opt for IGD 'cuz it gives a measure of the diversity and the convergence/accuracy of the PFs. However they do not state how many runs of each algorithm were made, which can hinder the correctness of these results. 

	In Zapotecas-Martinez 2016, they present an algorithm called MONSS (Multi-Objective Nonlinear Simplex Search approach). It uses the HV indicator and the two set coverage. Each algorithm was run 30 times. And they measured for each algorithm the average and std dev of each algorithm in the 30 runs.

	In Diaz-Manriquez 2016, they state the lack of standard approaches to compare different MOOAs.

-------------------------------------------------------------------

## E. Implementation and Competitive tools

1. Galapago's similarity graph represents the similarity between what? 

	A: Between the variables. 

2. Given that there are already out-of-the-box solvers available for each model, why didn't you use a mathematical modeling language? 

	A: Most of these modeling languages establish very strict syntax rules which must be followed and they instill some semantics to it that require more knowledge by the user (e.g., to know if the function is affine, quadratic or nonlinear). In addition, most of them do not support derivative-free solvers. Although JUMP did support derivative-free solvers (e.g., NLopt , this modeling language did not allowed for simulation-based functions and did not supported MOO. 

	Não excluimos o facto de virmos a usar, mas na altura decidimos enveredar por esta via, dadas as limitações encontradas (a nível de sintaxe, dos solvers que eram disponibilizados, e a nível também das funções que se podiam modelar com estas linguagens. No entnato, se surgirem novos solvers baseados em algoritmos derivative-free e as restrições sobre o tipo de expressões que se podem modelar com estas linguagens for relaxado, então poderemos pensar em optar por uma destas linguagens de modelação. (Não ser fundamentalista)

3. You state that you provide means to measure the performance of each algorithm. However to compare MOOAs some indicators require the existence of a reference set which is typically assumed to be the Pareto front. How do you compute this set during the benchmark? 

	A: The current implementation only computes the metrics in the end. We have defined functions that identify for each run the nondominated solutions. Because in general it is not possible to know the true Pareto Front we use the concept of a combined Pareto front which consists on optimal solutions obtained on each run. In the end of all the runs, the indicators are computed and outputed to a file. WHich can then be processed by a script which outputs them in tables and from which it is possible to extract some metrics.

-------------------------------------------------------------------

## Conclusions

1. What guidelines can we conclude about your work? Is it possible to know that the best algorithm for a structural optimization problem is X or Y? 

	A: Test. We conclude from our work that the best way to tackle these problems is to test the algorithms for a smaller amount of evaluations in order to understand which algorithm is the most promising. This is motivated by the lack of knowledge about the objective functions. Despite the existence of some rules of thumb regarding the number of dimensions and the type of behavior of each algorithm it is still impossible to know whether a specific algorithm should be applied to structural problems or daylight problems.

2. In your document you describe the impact of this dissertation for complex design problems. However, your case studies seem to be rather simple, presenting not more than 6 variables and not more than 2 objectives, merely subjected to bound constraints. How does this extrapolates for your conclusions? 

	A: My conclusions remain the same. The fact that these case studies are simple only strengthens this work. A dissertation has strict temporal deadlines which do not allow to study with the same level of detail more complicated case studies. If a single acoustic analysis for the C01 room at IST (square-based room) takes around 12h to complete, imagine how much it would take to assess an opera amphitheater or other more complicated case study. These conclusions are valuable and are applied to real case studies proposed by a small-scale architectural studio.

	Problemas de laboratório, mas também é pratica usual (E.g., zdt) as simplificações q fazemos são as mesmas q se fazem noutros casos. 
	É expectável que se tenha q fazer mta investigação para lidar c/ problemas c/ mtos objetivos e mtas vars em jogo.

-------------------------------------------------------------------
