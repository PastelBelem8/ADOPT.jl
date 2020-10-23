# Unit tests to module MOOIndicators

@testset "Pareto Front" begin

end
# 1. Test dominates function
#    - Wrong types
#    - Mismatching dimensions
@testset "Dominance relations" begin

end

# 2. Test isnondominated function
#    - Wrong types
#    - Mismatching dimensions

# 3. Test isparetoOptimal function

# 4. Test structure ParetoFront
#     - Creation of ParetoFront
#     - Order of Pareto Front
#     - isEmpty
#     - ...

# 5. Test `addPareto!` function

# 6. Test delete! function


# 7. Test addition of vectors to ParetoFront struct
#     - Pareto Optimal vector
#     - Pareto Optimal vector dominating 1 solution in PF
#     - Pareto Optimal vector dominating 2 solutions in PF.
#     - Pareto Optimal vector dominating all solutions in PF.
#     - Pareto non-optimal vector (PF should remain the same)
