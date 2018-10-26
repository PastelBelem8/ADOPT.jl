# Black Box optimization
using JuMP

function daylight_simulation(w, l)
    rkt_cmd = "racket C:\\Users\\catar\\Dropbox\\Work\\GAC-IST-WORK\\2018_InesPereira\\Multi-Objectivo\\PavPreto_1023.rkt"
    println("Running analysis for width=$w and length=$l.")
    res = chomp(read(`cmd /C $rkt_cmd $w $l`, String))
    print("Finished running analysis: sUDI= $(split(res, '\n')[end])")
    return parse(Float64, split(res, '\n')[end])
end

function gradient_daylight_simulation(w, l)
     println("Gradient w: $w; l: $l")
     return w+l
end

# using NLopt
# m = Model(solver = NLoptSolver(algorithm=:GN_CRS2_LM, maxeval=10)) # Not working
using Ipopt
m = Model(solver = IpoptSolver(resto_max_iter=10))
JuMP.register(m, :daylight_simulation, 2, daylight_simulation, gradient_daylight_simulation)

@variable(m, 1 <= x <= 4)
@variable(m, 1 <= y <= 17)
@NLobjective(m, Max,  daylight_simulation(x,y))

print(m)

solve(m)

println("Objective value: ", getobjectivevalue(m))
println("x = ", getvalue(x))
