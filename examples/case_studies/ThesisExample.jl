# Analytical Objective Functions
f1(x) = x^2
f2(x) = (x-2)^2

let var1 = RealVariable(-10, 10),
    obj1 = Objective(f1, :MIN),
    obj2 = Objective(f2, :MIN),
    model = Model([var1], [obj1, obj2])
  solve(NSGAII, model, max_evals=100)
end
