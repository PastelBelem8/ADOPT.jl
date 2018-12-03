module Metamodels

using ScikitLearnBase
# function fit!(model, X, y; kwargs...) end
# function predict(model, X; kwargs...) end

# ------------------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------------------
using DecisionTree: DecisionTreeRegressor, RandomForestRegressor, fit!, predict
export DecisionTreeRegressor, RandomForestRegressor

# ------------------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------------------
export LinearRegression
# FIXME - Replace this model by the one supported by ScikitLearn.jl.
# @Date: 15/11/2018 - Their definition of LinearRegression has bugs!
"""
    LinearRegression(; eltype=Float64, multi_output=nothing)

Linear regression. Supports both single-output and multiple-output regression.
Optimized for speed.

- `eltype`: the element type of the coefficient array. `Float64` is generally
best for numerical reasons.
- `multi_output`: for maximum efficiency, specify `multi_output=true/false`
"""
mutable struct LinearRegression{T<:Array,Y<:Number}
    coefs::T
    intercepts::Array{Y, 1}
    X::AbstractArray{Real, 2}
    y::AbstractArray{Real, 2}
    LinearRegression{T,Y}() where{T,Y} = new{T, Y}()
end

LinearRegression(; eltype=Float64, multi_output::Union{Nothing, Bool}=nothing) =
    multi_output === nothing ? LinearRegression{Array{eltype}, eltype}() : LinearRegression{Array{eltype, 2}, eltype}()

get_data(lr::LinearRegression, X, y) =
    isdefined(lr, :X) ? (hcat(lr.X, X), hcat(lr.y, y)) : (X, y)

ScikitLearnBase.fit!(lr::LinearRegression, X::AbstractArray{XT}, y::AbstractArray{yT}) where{XT, yT} =
    begin
        if XT == Float32 || yT == Float32
            warn("Regression on Float32 is prone to inaccuracy")
        end
        # Update Data
        X, y = get_data(lr, X, y)
        lr.X, lr.y = X, y

        # Compute weights and coefficients
        res = [ones(size(X, 2), 1) X'] \ y'
        lr.intercepts = res[1,:];
        lr.coefs = res[2:end,:];
        lr
    end
ScikitLearnBase.predict(lr::LinearRegression, X) =
    lr.coefs' * X .+ lr.intercepts

# ------------------------------------------------------------------------
# Support Vector Regression (SVR)
# ------------------------------------------------------------------------
using LIBSVM: NuSVR, EpsilonSVR, fit!, predict
using LIBSVM: Kernel.Linear, Kernel.RadialBasis, Kernel.Polynomial,
              Kernel.Sigmoid,Kernel.Precomputed

export NuSVR, EpsilonSVR
export Linear, RadialBasis, Polynomial, Sigmoid, Precomputed

# ------------------------------------------------------------------------
# Multi-Layer Perceptron Regression
# ------------------------------------------------------------------------
include("MLPRegressor.jl")

export  ADADelta, ADAGrad, ADAM, AdaMax, ADAMW, AMSGrad,
        Momentum, NADAM, Nesterov, RMSProp, SGD

export MLPRegressor

# ------------------------------------------------------------------------
# Gaussian Processes
# ------------------------------------------------------------------------
using GaussianProcesses: GPE
# Means
using GaussianProcesses: MeanConst, MeanLin, MeanPoly, MeanZero
using GaussianProcesses: SumMean, ProdMean
# Kernels
using GaussianProcesses: Const,
                         LinArd, LinIso,
                         Matern,
                         Mat12Iso, Mat32Iso, Mat52Iso,
                         Mat12Ard, Mat32Ard, Mat52Ard,
                         Noise,
                         Poly,
                         RQ, RQIso, RQArd,
                         SE, SEArd, SEIso,
                         # Composite Kernels
                         ProdKernel,
                         SumKernel
# Likelihood
using GaussianProcesses: BernLik,
                         BinLik,
                         ExpLik,
                         GaussLik,
                         PoisLik,
                         StuTLik

ScikitLearnBase.fit!(gp::GPE, X::AbstractMatrix, y::AbstractVector) = GaussianProcesses.fit!(gp, X', y)
ScikitLearnBase.predict(gp::GPE, X::AbstractMatrix; eval_MSE::Bool=false) =
    begin
        optimise!(gp)
        mu, Sigma = GaussianProcesses.predict_y(gp, X'; full_cov=false)
        if eval_MSE
           return mu, Sigma
        else
           return mu
        end
    end

export GPE
export MeanConst, MeanLin, MeanPoly, MeanZero, SumMean, ProdMean
export Const, LinArd, LinIso, Matern, Mat12Iso, Mat32Iso, Mat52Iso,
       Mat12Ard, Mat32Ard, Mat52Ard, Noise, Poly, RQ, RQIso, RQArd,
       SE, SEArd, SEIso, ProdKernel, SumKernel
export BernLik, BinLik, ExpLik, GaussLik, PoisLik, StuTLik

end # Module
