module SurrogateModelsTests

using Test
import MscThesis.SurrogateModels: LinearRegression, fit!, predict
import MscThesis.SurrogateModels: MLPRegressor, gen_batches

@testset "Linear Regression Tests" begin
    close_enough(x0, x1, tol=1e-14) = abs(x0 - x1) <= tol ? true : false
    @testset "Single-Target Keyword Tests" begin

        @testset "One-independent variable Tests" begin
            # Matrices should be n_features x n_samples
            X = [1 2 3 4 5]
            y = identity.(X)

            @test begin
                y_pred = predict(fit!(LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = predict(
                fit!(LinearRegression(multi_output=false), X, y),
                X)
                all(map(close_enough, y_pred, y))
            end
         end

         @testset "Two-independent variables Tests" begin
             X = [1 2 3 4 5 ;
                  2 4 6 8 10]
             y = mapslices(x -> x[1] - x[2], X, dims=1)

             @test begin
                 y_pred = predict(fit!(LinearRegression(), X, y), X)
                 all(map(close_enough, y_pred, y))
             end
             @test begin
                 y_pred = predict(
                 fit!(LinearRegression(multi_output=false), X, y),
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
                y_pred = predict(fit!(LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = predict(
                        fit!(LinearRegression(multi_output=true), X, y),
                        X)
                all(map(close_enough, y_pred, y))
            end
        end

        @testset "Two-independent variable Tests" begin
            X = [1 2 3 4 5 ;
                 2 4 6 8 10]
            y = mapslices(x -> [x[1] + x[2], x[1] - x[2]], X, dims=1)

            @test begin
                y_pred = predict(fit!(LinearRegression(), X, y), X)
                all(map(close_enough, y_pred, y))
            end
            @test begin
                y_pred = predict(
                        fit!(LinearRegression(multi_output=true), X, y),
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
#         # fit!(reg1, X1, y1)
#         # y_pred1 = predict(reg1, X1)
#         #
#         # using Plots
#         # scatter(X1', y1', title="MLP Regression", label="Data")
#         # plot!(X1', y_pred1', linewidth=6, label="MLP model (ReLU)")
#
#     end
# end
end # module
