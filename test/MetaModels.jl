module MetaModelsTests

using Test
import MscThesis.Metamodels
# import MscThesis.Metamodels: MLPRegressor, gen_batches
@testset "Linear Regression Tests" begin
    close_enough(x0, x1, tol=1e-14) = abs(x0 - x1) <= tol ? true : false
    @testset "Single-Target Keyword Tests" begin

        @testset "One-independent variable Tests" begin
            # Matrices should be n_features x n_samples
            X = [1 2 3 4 5]
            y = identity.(X)

            @test begin
                y_pred = Metamodels.predict(Metamodels.fit!(Metamodels.LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = Metamodels.predict(
                Metamodels.fit!(Metamodels.LinearRegression(multi_output=false), X, y),
                X)
                all(map(close_enough, y_pred, y))
            end
         end

         @testset "Two-independent variables Tests" begin
             X = [1 2 3 4 5 ;
                  2 4 6 8 10]
             y = mapslices(x -> x[1] - x[2], X, dims=1)

             @test begin
                 y_pred = Metamodels.predict(Metamodels.fit!(Metamodels.LinearRegression(), X, y), X)
                 all(map(close_enough, y_pred, y))
             end
             @test begin
                 y_pred = Metamodels.predict(
                 Metamodels.fit!(Metamodels.LinearRegression(multi_output=false), X, y),
                 X)
                 all(map(close_enough, y_pred, y))
             end
         end
    end

    @testset "Multi-Target Keyword Tests" begin
        # Matrices should be n_features x n_samples
        @testset "One-independent variable Tests" begin
            X = [1 2 3 4 5]
            y = [1  2  3  4  5;
                -1 -2 -3 -4 -5]
            @test begin
                y_pred = Metamodels.predict(Metamodels.fit!(Metamodels.LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = Metamodels.predict(
                        Metamodels.fit!(Metamodels.LinearRegression(multi_output=true), X, y),
                        X)
                all(map(close_enough, y_pred, y))
            end
        end

        @testset "Two-independent variable Tests" begin
            X = [1 2 3 4 5 ;
                 2 4 6 8 10]
            y = mapslices(x -> [x[1] + x[2], x[1] - x[2]], X, dims=1)

            @test begin
                y_pred = Metamodels.predict(Metamodels.fit!(Metamodels.LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = Metamodels.predict(
                        Metamodels.fit!(Metamodels.LinearRegression(multi_output=true), X, y),
                        X)
                all(map(close_enough, y_pred, y))
            end
        end
    end
end



# @testset "Multi-Layer Perceptron Regression Tests" begin
#     @testset "Auxiliar methods" begin
#         @testset "gen_batches" begin
#             X = [3 9 5 3 2 1 5 ;
#                  7 8 9 1 9 1 2 ;]
#             y = [1 2 3 4 5 6 7 8;]
#
#         end
#     end
#
#     @testset "MLPRegressor" begin
#
#         # X1 = [1 2 3 4 5 6 7 8 9 10;]
#         # y1 = vcat(map(identity,X1), map(x -> -x |> identity, X1))
#         #
#         # solver = ADAM()
#         # reg1 = MLPRegressor((1, 100, 2), solver=solver, batch_size=5, batch_shuffle=true)
#         # Metamodels.fit!(reg1, X1, y1)
#         # y_pred1 = Metamodels.predict(reg1, X1)
#         #
#         # using Plots
#         # scatter(X1', y1', title="MLP Regression", label="Data")
#         # plot!(X1', y_pred1', linewidth=6, label="MLP model (ReLU)")
#
#     end
# end


#= Tests
X1 = reshape(Vector(1:1000), (1, 1000))
y1 = vcat(map(identity,X1), map(x -> -x |> identity, X1))

λ=0.01
solver = ADAM()
reg1 = MLPRegressor((1, 100, 100, 2), solver=solver, batch_size=75, epochs=30, λ=λ)

Metamodels.fit!(reg1, X1, y1)

using Plots
plot(Vector(1:length(reg1.early_stopping.validation_losses)), reg1.early_stopping.validation_losses, yscale=:log10, label="Validation Losses")
plot!(Vector(1:length(reg1.losses)), reg1.losses, yscale=:log10, label="Training Losses")

y_pred1 = Metamodels.predict(reg1, X1)

println("Minimum training losses: $(minimum(reg1.losses))")
println("Minimum validation losses: $(minimum(reg1.early_stopping.validation_losses))")

using Plots
scatter(X1', y1', title="MLP Regression", label="Data")
plot!(X1', y_pred1', linewidth=6, label="MLP model (ReLU)")
=#


#= Tests
X2 = [0 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30;
      0 -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 -29 -30;]
y2 = mapslices(x -> x[1] * x[2], X2, dims=1)

λ = 0.03
solver = ADAM(0.3, decay=0.25)
reg2 = MLPRegressor((2, 100, 100, 1), solver=solver, epochs=1200, batch_size=5, λ=λ)
Metamodels.fit!(reg2, X2, y2)
y_pred2 = Metamodels.predict(reg2, X2)

using Plots
scatter(y2', X2', title="MLP Regression", label="Data")
plot!(y_pred2', X2', linewidth=6, label="MLP model (ReLU)")
scatter(X2', y2', title="MLP Regression", label="Data")
plot!(X2', y_pred2', linewidth=6, label="MLP model (ReLU)")

minimum(reg2.losses)

plot(Vector(1:length(reg1.losses)), reg1.losses)
plot!(Vector(1:length(reg1.early_stopping.validation_losses)), reg1.early_stopping.validation_losses)
plot!(Vector(1:length(reg2.losses)), reg2.losses)
plot!(Vector(1:length(reg2.early_stopping.validation_losses)), reg2.early_stopping.validation_losses)
=#

#= Tests
X3 = reshape(Vector(1:1000), (1, 1000))
y3 = vcat(map(identity,X1), map(x -> -x^2 |> identity, X1))

λ=0.01
solver = ADAM()
reg3 = MLPRegressor((1, 100, 100,50,100, 2), solver=solver, batch_size=100, epochs=30, λ=λ)

Metamodels.fit!(reg3, X3, y3)

using Plots
plot(Vector(1:length(reg3.early_stopping.validation_losses)), reg3.early_stopping.validation_losses, yscale=:log10, label="Validation Losses")
plot!(Vector(1:length(reg3.losses)), reg3.losses, yscale=:log10, label="Training Losses")

y_pred3 = Metamodels.predict(reg3, X1)

println("Minimum training losses: $(minimum(reg3.losses))")
println("Minimum validation losses: $(minimum(reg3.early_stopping.validation_losses))")

using Plots
scatter(X3', y3', title="MLP Regression", label="Data", markersize=2)
plot!(X3', y_pred3', label="MLP model (ReLU)", linewidth=3)
=#


#= Tests # No Validation-Based Early Stopping (EarlyStopping should be registering training data)
X4 = reshape(Vector(1:1000), (1, 1000))
y4 = vcat(map(x -> -x^3,X4), map(x -> -x^2, X4))

λ=0.01
solver = ADAM()
reg4 = MLPRegressor((1, 100, 100,50,100, 2), solver=solver, batch_size=100, epochs=300, λ=λ, val_fraction=0)

Metamodels.fit!(reg4, X4, y4)

using Plots
plot(Vector(1:length(reg4.early_stopping.validation_losses)), reg4.early_stopping.validation_losses, yscale=:log10, label="Validation Losses")
plot!(Vector(1:length(reg4.losses)), reg4.losses, yscale=:log10, label="Training Losses")

y_pred4 = Metamodels.predict(reg4, X4)

println("Minimum training losses: $(minimum(reg4.losses))")
println("Minimum validation losses: $(minimum(reg4.early_stopping.validation_losses))")

using Plots
scatter(X4', y4', title="MLP Regression", label="Data", markersize=2)
plot!(X4', y_pred4', label="MLP model (ReLU)", linewidth=3)
=#


end # module
