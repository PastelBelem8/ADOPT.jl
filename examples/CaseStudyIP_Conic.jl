using Main.MscThesis
using Main.MscThesis.Platypus
# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
scale(x, min, step=0.1) = min + step*x

rectangular_cost_function(x) = let
    p1 = scale(x[1], 1.5) * scale(x[2], 6.5) * 185
    p2 = (scale(x[1], 1.5) + scale(x[2], 6.5)) * 3 * 80
    p1 + p2
end

# x = [nskylights, max_dist, center_y, radius_m, radius_M, material]
conic_cost_function(x; height=1.5) = let
    nskylights = x[1]
    radius_m = scale(x[4], 0.2)
    radius_M = scale(x[5], 0.3)
    println("conic cost function: $x")
    s = √(√(height) + √(radius_M - radius_m))
    p1 = π * (radius_M + radius_m) * s * 80
    p2 = π * √radius_m * 185
    (p1 + p2) * nskylights
end

daylight_function(x) = let
    nskylights = x[1]
    d = scale(x[2], 0.14, 0.01)
    center_y = scale(x[3], 0.44, 0.01)
    radius_m = scale(x[4], 0.2)
    radius_M = scale(x[5], 0.3)
    material = x[6]
    cmd = `cmd /C racket $(@__DIR__)examples/CaseStudyIP/PavPreto_190316.rkt $(nskylights) $(d) $(center_y) $(radius_m) $(radius_M) $(material)`
    output = chomp(Base.read(cmd, String))
    output = split(output, '\n')[end]
    parse(Float64, output)
end


# x = [1-nskylights, 2-max_dist, 3- center_y, 4-radius_m, 5-radius_M, 6-material]
let nskylights = IntVariable(2, 8)
    max_dist = IntVariable(0, 2216)
    center_y = IntVariable(0, 205)
    radius_m = IntVariable(0, 8)
    radius_M = IntVariable(0, 8)
    material = IntVariable(0, 2)
    vars = [nskylights, max_dist, center_y, radius_m, radius_M, material]
    objs = [Objective(conic_cost_function, :MIN),
            Objective(conic_cost_function, :MIN)]
            # Objective((x) -> -1 * daylight_function(x), :MIN)]
    # Constraint 1: Aims at preventing the bottom skylight from crossing the wall.
    # expression: center_y >= radius_M
    constraint_1(x) = (scale(x[3], 0.44, 0.01) - 0.14) - scale(x[5], 0.3) # >= 0
    # Constraint 2: Aims at preventing intersections between the different skylights
    # expression: (nskylights-1) * radius_M * 2 <= d
    constraint_2(x) = (x[1] - 1) * (scale(x[5], 0.3) * 2) - scale(x[2], 0.14, 0.01) # <= 0
    # Constraint 3: Aims at preventing the larger radius from being smaller than the smaller one.
    # expression: radius_M > radius_m
    constraint_3(x) = scale(x[5], 0.3) - scale(x[4], 0.2) # > 0
    # Constraint 4: Aims at preventing the total extension of the conic skylights larger than the building's length
    # expression: center_y + d + radius_M <= length
    constraint_4(x) = scale(x[3], 0.44, 0.01) + scale(x[2], 0.14, 0.01) + scale(x[5], 0.3) - 22.16 # <= 0
    constrs = [ Constraint(constraint_1, >=),
                Constraint(constraint_2, <=),
                Constraint(constraint_3, >),
                Constraint(constraint_4, <=)]
    model = Model(vars, objs, constrs)
    maxevals = 200
    nparticles = 10
    EAs_params = Dict(:population_size => nparticles)
  benchmark(nruns=3, algorithms=[(NSGAII, EAs_params)], problem=model, max_evals=maxevals)
end

#=
constraint_1(x) = (scale(x[3], 0.44, 0.01) - 0.14) - scale(x[5], 0.3) # >= 0
constraint_2(x) = (x[1] - 1) * (scale(x[5], 0.3) * 2) - scale(x[2], 0.14, 0.01) # <= 0
constraint_3(x) = scale(x[5], 0.3) - scale(x[4], 0.2) # > 0
constraint_4(x) = scale(x[3], 0.44, 0.01) + scale(x[2], 0.14, 0.01) + scale(x[5], 0.3) - 22.16 # <= 0
constrs = [ Constraint(constraint_1, >=),
            Constraint(constraint_2, <=),
            Constraint(constraint_3, >),
            Constraint(constraint_4, <=)]
x = [2, 542, 54, 1, 1, 1] # [2, 639, 69, 2, 1, 0]

println("===== CONSTRAINED ====")
cnstrs_values = map(c -> evaluate(c, x), constrs)

# Compute penalty
cnstrs_penalty = Main.MscThesis.penalty(constrs, cnstrs_values)
feasible = iszero(cnstrs_penalty)
println(">>> Variables: $(x) \n>>>is_feasible: $(feasible)")
objs_values, objs_time = feasible ? eval_objectives() : (zeros(length(objs)), -1 * ones(length(objs)))
println(">>> Expected: $(objs_values) <<<<< ")
write_result("evaluate", time()-start_time, cnstrs_time, objs_time, vars,
            cnstrs_values, cnstrs_penalty, feasible, objs_values)
println(">>> Written! <<<<< ")

Solution(vars, objs_values, cnstrs_values, cnstrs_penalty, feasible, true)


=#
