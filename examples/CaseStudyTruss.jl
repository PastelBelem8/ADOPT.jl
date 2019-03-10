using Khepri
# -------------------------------------------------------------------------
#                                 Primitives
# -------------------------------------------------------------------------
fixed_xyz_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=create_node_support("SupportA", ux=true, uy=true, uz=true))
fixed_z_truss_node_family =
  truss_node_family_element(default_truss_node_family(),
                            support=create_node_support("SupportB", ux=false, uy=false, uz=true))

default_truss_bar_family(
  truss_bar_family_element(
    default_truss_bar_family(),
    material=[
      "ElasticIsotropic",   # name
      Khepri.I_MT_STEEL,    # Type
      "Steel",              # Name
      "I'm really steel",   # Nuance
      210000000000.0,       # E
      0.3,                  # NU
      81000000000.0,        # Kirchoff
      77010.0,              # RO
      1.2e-05,              # LX
      0.04,                 # DumpCoef
      235000000.0,          # RE
      360000000.0],         # RT
    section=[
      "Tube",               #name
      "ElasticIsotropic",   #material_name
      false,                #iswood
      [(true,               #solid?
        0.1,                #diameter
        0.01)]]))           #thickness

no_trelica(p) = truss_node(p)

fixed_xyz_no_trelica(p) = truss_node(p, fixed_xyz_truss_node_family)

fixed_z_no_trelica(p) = truss_node(p, fixed_z_truss_node_family)

barra_trelica(p0, p1) = truss_bar(p0, p1)

# -------------------------------------------------------------------------
#                                 Treliça
# -------------------------------------------------------------------------
nos_trelica(ps) =
  for p in ps
    no_trelica(p)
  end

fixed_nos_trelica(ps) =
  begin
    fixed_xyz_no_trelica(ps[1])
    nos_trelica(ps[2:end-1])
    fixed_xyz_no_trelica(ps[end])
  end

barras_trelica(ps, qs) =
  for (p, q) in zip(ps, qs)
    barra_trelica(p, q)
  end

trelica_espacial(curvas, f, isfirst=true) = let
  (as, bs, cs) = (curvas[1], curvas[2], curvas[3])
  isfirst ? fixed_nos_trelica(as) : nos_trelica(as)
  nos_trelica(bs)
  curvas[4:end] == [] ?
    (fixed_nos_trelica(cs);
     barras_trelica(cs[2:end], cs)) :
    (trelica_espacial(curvas[3:end], f, false);
     barras_trelica(bs, curvas[4]))
  barras_trelica(as, cs)
  barras_trelica(bs, as)
  barras_trelica(bs, cs)
  barras_trelica(bs, as[2:end])
  barras_trelica(bs, cs[2:end])
  barras_trelica(as[2:end], as)
  barras_trelica(bs[2:end], bs)
end

# -------------------------------------------------------------------------
#                              Parameters
# -------------------------------------------------------------------------
attractors = Parameter([xyz(5, 5, 5)])

# Attractors will create perturbation in the truss nodes in its vicinities
affect_radius(r, p) =
  r*(1+0.5*1.4^(-min(map(attractor -> distance(p, attractor), attractors())...)))

pontos_arco(p, r, fi, psi0, psi1, dpsi) =
  psi0 > psi1 ?
    [] :
    vcat([p+vsph(affect_radius(r, p+vsph(r, fi, psi0)), fi, psi0)],
         pontos_arco(p, r, fi, psi0+dpsi, psi1, dpsi))

coordenadas_trelica_ondulada(p, rac, rb, l, fi, psi0, psi1, dpsi, alfa0, alfa1, d_alfa, d_r) =
  alfa0 >= alfa1 ?
    [pontos_arco(p+vpol(l/2.0, fi-pi/2), rac+d_r*sin(alfa0), fi, psi0, psi1, dpsi)] :
    vcat([pontos_arco(p+vpol(l/2.0, fi-pi/2), rac+d_r*sin(alfa0), fi, psi0, psi1, dpsi)],
         vcat([pontos_arco(p, rb+d_r*sin(alfa0), fi, psi0+dpsi/2, psi1-dpsi/2, dpsi)],
              coordenadas_trelica_ondulada(p+vpol(l, fi+pi/2), rac, rb, l, fi, psi0, psi1, dpsi, alfa0+d_alfa, alfa1, d_alfa, d_r)))

trelica_ondulada(p, rac, rb, l, n, fi, psi0, psi1, alfa0, alfa1, d_alfa, d_r) =
  trelica_espacial(
    coordenadas_trelica_ondulada(p, rac, rb, l, fi, psi0, psi1, (psi1-psi0)/n, alfa0, alfa1, d_alfa, d_r),
    panel)

# -------------------------------------------------------------------------
#                       Truss Algorithmic Model
# -------------------------------------------------------------------------
spiked_truss(α1, y1, α2, y2, α3, y3) =
  Khepri.with(attractors, [ sph(10, 0, α1)+vy(y1),
                            sph(10, 0, α2)+vy(y2),
                            sph(10, 0, α3)+vy(y3)]) do
    delete_all_shapes()
    trelica_ondulada(xyz(0, 0, 0),10,9,1.0,10,0,-pi/2,pi/2,0,4*pi,pi/8,0.1)
  end

# -------------------------------------------------------------------------
#                              Configurations
# -------------------------------------------------------------------------
# Dependencies
using LinearAlgebra
using Combinatorics

# Set the analysis backend
backend(robot)
project_kind(Khepri.I_PT_SHELL)

# -------------------------------------------------------------------------
#                             Objectives
# -------------------------------------------------------------------------
# Minimize the maximum displacement
spiked_truss_displacement(α1, y1, α2, y2, α3, y3) =
  Khepri.with(attractors, [sph(10, 0, α1)+vy(y1),
                    sph(10, 0, α2)+vy(y2),
                    sph(10, 0, α3)+vy(y3)]) do
    delete_all_shapes()
    new_robot_analysis(
      ()-> trelica_ondulada(xyz(0, 0, 0),10,9,1.0,10,0,-pi/2,pi/2,0,4*pi,pi/8,0.1),
      vxyz(-50000.0, 0.0, 0.0)) do results
        let displs = displacements(nodes(results))
            node_displ = node ->
              norm(node_displacement_vector(displs, node.id, Khepri.I_LRT_NODE_DISPLACEMENT))
            disps = maximum([node_displ(node) for node in values(added_nodes())])
          disps
      end
    end
end

# Measure of irregularity: Minimize the euclidean distance
spiked_truss_style(α1, y1, α2, y2, α3, y3) = begin
  Khepri.with(attractors, [sph(10, 0, α1)+vy(y1),
                    sph(10, 0, α2)+vy(y2),
                    sph(10, 0, α3)+vy(y3)]) do
    sum(distance(p0, p1) for (p0, p1) in combinations(attractors(), 2))
  end
end

# -------------------------------------------------------------------------
#                             Optimization
# -------------------------------------------------------------------------
using Main.MscThesis

# Angle, longitudinal position, ...
vars = [RealVariable(-π, π), RealVariable(0, 4π),
        RealVariable(-π, π), RealVariable(0, 4π),
        RealVariable(-π, π), RealVariable(0, 4π)]

objs = [Objective(x -> spiked_truss_displacement(x...), 1, :MIN), # 20s - 140s
        Objective(x -> spiked_truss_style(x...), :MIN)] # 0.0s

problem = Model(vars, objs)

# Step 2. Define the Solver
using Main.MscThesis.Platypus
using Main.MscThesis.Sampling
using Main.MscThesis.ScikitLearnModels

# -------------------------------------------------------------------------
#                   1ST TEST - 1x surrogate - 2 objs
# -------------------------------------------------------------------------
iter = 10
nparticles = 50 # > 6*(6+1)
nevals_mtsolver = nparticles * iter
maxevals= 225

# Metaheuristic
ea_solver() = let
  params = Dict(:population_size => nparticles)
  Main.MscThesis.PlatypusSolver(NSGAII, max_eval=nevals_mtsolver, algorithm_params=params, nondominated_only=true)
end

pso_solver() = let
  params = Dict(:leader_size => nparticles,
                :swarm_size => nparticles,
                :max_iterations => nparticles,
                :mutation_probability => 0.3,
                :mutation_perturbation => 0.5)
  PlatypusSolver(SMPSO, max_eval=nevals_mtsolver, algorithm_params=params, nondominated_only=true)
end

sampling_solver() = let
  params = Dict(:sampling_function => randomMC,
                :nsamples => nevals_mtsolver)
  SamplingSolver(;algorithm_params=params, max_eval=nevals_mtsolver, nondominated_only=true)
end

# Meta Solver
meta_solver(metamodel, solver, X, y) = let
  params = Dict(:X => X, :y => y)
  surrogate = Surrogate(  metamodel, objectives=objs, creation_f=sk_fit!,
                          update_f=sk_fit!, evaluation_f=sk_predict)
  MetaSolver(solver; surrogates=[surrogate], max_eval=225, sampling_params=params, nondominated_only=true)
end

# Test 1 - GPR
gpr_1 = (X, y) -> meta_solver(GaussianProcessRegressor(), pso_solver(), X, y)
gpr_2 = (X, y) -> meta_solver(GaussianProcessRegressor(), ea_solver(), X, y)
gpr_3 = (X, y) -> meta_solver(GaussianProcessRegressor(), sampling_solver(), X, y)
# Test 2 - Random Forest
random_forest_1 = (X, y) -> meta_solver(RandomForestRegressor(), pso_solver(), X, y)
random_forest_2 = (X, y) -> meta_solver(RandomForestRegressor(), ea_solver(), X, y)
random_forest_3 = (X, y) -> meta_solver(RandomForestRegressor(), sampling_solver(), X, y)

# Test 3 - SVR
svr_1 = (X, y) -> meta_solver(SVR(), pso_solver(), X, y)
svr_2 = (X, y) -> meta_solver(SVR(), ea_solver(), X, y)
svr_3 = (X, y) -> meta_solver(SVR(), sampling_solver(), X, y)

using DelimitedFiles
readdata(filename) = let
  data = readdlm(filename, ',', Float64, '\n'; header=false)
  data[:,1:6]', data[:, 7:8]'
end

X1, y1 = readdata("$(@__DIR__)/CaseStudyTruss/truss_sample1.csv")
X2, y2 = readdata("$(@__DIR__)/CaseStudyTruss/truss_sample2.csv")
X3, y3 = readdata("$(@__DIR__)/CaseStudyTruss/truss_sample3.csv")

# Depois de testar estes tres e ver se há diferença entre os tres, logo se ve se faz sentido fazer tantos testes, destes...
# Main.MscThesis.solvers_benchmark(nruns=3,
#                                  Xs=[X1, X2, X3],
#                                  ys=[y1, y2, y3],
#                                  solvers=[gpr_1, gpr_2, gpr_3],
#                                  problem=problem,
#                                  max_evals=maxevals)

solve(gpr_1(X1, y1), problem)
# Fica a faltar
# 1. Outros testes (random_forest, svr) - difernetes variantes dependem dos testes de GPR
# 2. 2x surrogates - 2 objs



# Step 1. Define the algorithms to be run:
#   2 tests: 1x surrogate - 2 objs
#            2x surrogates - 2 objs
#   Algorithms: GP + PSO, GP + NSGAII, GP + LHS
#               RT + PSO, RT + NSGAII, RT + LHS
#               SVR + PSO, SVR + NSGAII, SVR + LHS
# Step 2. Create files with initial samples: n_init = 100
# Step 3. Each algorithm will run 3x 115 analysis (the other 100 will be read from the files)
# Step 4. Create the routines
