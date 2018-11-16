module SurrogateModels

using ScikitLearnBase
import ScikitLearnBase: @declare_hyperparameters, fit!, predict
export fit!, predict

# ------------------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------------------
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor
using DecisionTree: fit!, predict
export DecisionTreeRegressor, RandomForestRegressor

# ------------------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------------------
# FIXME - Replace this model by the one supported by ScikitLearn.jl.
# @Date: 15/11/2018 - Their definition of LinearRegression has bugs!
mutable struct LinearRegression{T<:Array,Y<:Number}
    coefs::T
    intercepts::Array{Y, 1}
    LinearRegression{T,Y}() where{T,Y} = new{T, Y}()
end

"""    LinearRegression(; eltype=Float64, multi_output=nothing)

Linear regression. Supports both single-output and multiple-output regression.
Optimized for speed.

- `eltype`: the element type of the coefficient array. `Float64` is generally
best for numerical reasons.
- `multi_output`: for maximum efficiency, specify `multi_output=true/false` """
LinearRegression(; eltype=Float64, multi_output::Union{Nothing, Bool}=nothing) =
    multi_output === nothing ?  LinearRegression{Array{eltype}, eltype}() :
                                LinearRegression{Array{eltype, 2}, eltype}()

@declare_hyperparameters(LinearRegression, Symbol[])

function ScikitLearnBase.fit!(lr::LinearRegression, X::AbstractArray{XT},
                              y::AbstractArray{yT}) where {XT, yT}
    if XT == Float32 || yT == Float32
        warn("Regression on Float32 is prone to inaccuracy")
    end
    results = [ones(size(X, 2), 1) X'] \ y'
    lr.intercepts = results[1,:];
    lr.coefs = results[2:end,:];
    lr
end

ScikitLearnBase.predict(lr::LinearRegression, X) = lr.coefs' * X .+ lr.intercepts


# ------------------------------------------------------------------------
# Support Vector Regression (SVR)
# ------------------------------------------------------------------------
using LIBSVM: SVC, NuSVC, NuSVR, EpsilonSVR, LinearSVC
using LIBSVM: Kernel.Linear, Kernel.RadialBasis, Kernel.Polynomial,
              Kernel.Sigmoid,Kernel.Precomputed
using LIBSVM: fit!, predict

export SVC, NuSVC, NuSVR, EpsilonSVR, LinearSVC
export Linear, RadialBasis, Polynomial, Sigmoid, Precomputed

# ------------------------------------------------------------------------
# Multi-Layer Perceptron Regression
# ------------------------------------------------------------------------


end # Module
