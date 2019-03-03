module ScikitLearnModels

using PyCall
using MacroTools

const scikitlearn = PyNULL()

function __init__()
    copy!(scikitlearn, pyimport_conda("sklearn", "scikit-learn"))
    version = VersionNumber(scikitlearn[:__version__])
    @info("Your Python's scikit-learn has version $version.")
end

#  ------------------------- Configurations -------------------------
""" Like `pyimport` but gives a more informative error message """
importpy(name::AbstractString) =  try pyimport(name)
    catch e
        if isa(e, PyCall.PyError)
            error("This scikit-learn functionality ($name) requires installing the Python scikit-learn library.")
        else
            rethrow()
        end
    end

# Maybe these can be replaced by direct access to scikitlearn variable ?
sklearn() = scikitLearn         # importpy("sklearn")
sk_base() = scikitlearn[:base]  # importpy("sklearn.base")

clone(py_model::PyObject) = sklearn()[:clone](py_model, safe=true)

is_regressor(py_model::PyObject) = sk_base()[:is_regressor](py_model)

get_classes(py_estimator::PyObject) = py_estimator[:classes_]
get_components(py_estimator::PyObject) = py_estimator[:components_]

#  ------------------------- Julia -> Python -------------------------
# Julia => Python (methods)
api_map = Dict(:decision_function => :decision_function,
               :fit! => :fit,
               :fit_predict! => :fit_predict,
               :fit_transform! => :fit_transform,
               :get_feature_names => :get_feature_names,
               :get_params => :get_params,
               :inverse_transform => :inverse_transform,
               :predict => :predict,
               :predict_proba => :predict_proba,
               :predict_log_proba => :predict_log_proba,
               :partial_fit! => :partial_fit,
               :score_samples => :score_samples,
               :sample => :sample,
               :score => :score,
               :transform => :transform,
               :set_params! => :set_params)

# PyCall does not always convert everything back into a Julia value,
# unfortunately, so we have some post-evaluation logic. These should be fixed
# in PyCall.jl
tweak_rval(x) = x
function tweak_rval(x::PyObject)
   numpy = importpy("numpy")
   if pyisinstance(x, numpy[:ndarray]) && length(x[:shape]) == 1
       return collect(x)
   else
       x
   end
end

for (jl_fun, py_fun) in api_map
   @eval $jl_fun(py_model::PyObject, args...; kwargs...) =
       tweak_rval(py_model[$(Expr(:quote, py_fun))](args...; kwargs...))
end

macro sk_op(class, operation, params...)
    get_name = (param) -> isa(param, Expr) ? param.args[1] : param
    get_type = (param) -> isa(param, Expr) ? param.args[2] : nothing

    param_names = map(get_name, params)
    param_types = map(get_type, params)

    op_params = map((name, typ) -> typ == nothing ? (:t) : (:(t::$(typ))),
                        param_names, param_types)

    quote
        $(esc(operation))(py_model::PyObject, $(op_params...); kwargs...) =
            tweak_rval(py_model[$(Expr(:quote, operation))]($(op_params...); kwargs...))
    end
end

""" `predict_nc(model, X)` calls predict on the Python `model`, but returns
the result as a `PyArray`, which is more efficient than the usual path. See
PyCall.jl """
predict_nc(model::PyObject, X) = pycall(model[:predict], PyArray, X)
predict_nc(model::Any, X) = predict(model, X) # default

# Get symbols inside an expression
symbols_in(e::Expr) = union(symbols_in(e.head), map(symbols_in, e.args)...)
symbols_in(e::Symbol) = Set([e])
symbols_in(::Any) = Set()

# Import models from Python library
"""
@sk_import imports models from the Python version of scikit-learn. For instance, the
Julia equivalent of
`from sklearn.linear_model import LinearRegression, LogicisticRegression` is:
    @sk_import linear_model: (LinearRegression, LogisticRegression)
    model = fit!(LinearRegression(), X, y)
"""
macro sk_import(expr)
    # 1. Syntax verification
    @assert @capture(expr, mod_:toImport_) "`@sk_import` syntax error. Try something like: @sk_import linear_model: (LinearRegression, LogisticRegression)"
    if :sklearn in symbols_in(expr)
        error("Bad @sk_import: please remove `sklearn.` (it is implicit)")
    end
    # 2. Get the models to import
    if isa(toImport, Symbol)
        members = [toImport]
    else
        @assert @capture(toImport, ((members__),)) "Bad @sk_import statement"
    end
    # 2. Create the code to be expanded
    mod_string = "sklearn.$mod"
    quote
        mod_obj = pyimport($mod_string)
        $([:(const $(esc(w)) = mod_obj[$(Expr(:quote, w))]) for w in members]...)
        $([:(export $(esc(w))) for w in members]...)
    end
end


#  ------------------------- Sklearn Operations -------------------------
using LinearAlgebra
# FIXME - Hack to make this work. !! PLEASE DO NOT MIMICK! THIS IS AWFUL! :(
PyObject(x::LinearAlgebra.Adjoint) = PyObject(copy(x))
PyObject(x::Transpose) = PyObject(copy(x))

# FIXME I did it to allow the convresion of the format used in the MscThesis (nfeatures x nsamples)
# to (nsamples x nfeatures), without compromising the whole optimization framework.
sk_fit!(model, X, y; kwargs...) = fit!(model, X', y'; kwargs...)
sk_predict(model, X; kwargs...) = predict(model, X'; kwargs...)
export sk_fit!, sk_predict

#  ------------------------- Configurations -------------------------

# Linear models
@sk_import linear_model: (LinearRegression, LogisticRegression, BayesianRidge, ARDRegression)

# Decision Tree
@sk_import tree: (DecisionTreeRegressor, ExtraTreeRegressor)
# @sk_op DecisionTreeRegressor decision_path X

@sk_import ensemble: (RandomForestRegressor, ExtraTreesRegressor)
# @sk_op RandomForestRegressor decision_path X

@sk_import neural_network: (MLPRegressor, BernoulliRBM)
@sk_import svm: (SVR, NuSVR, LinearSVR)

@sk_import gaussian_process: (GaussianProcessRegressor,)
@sk_import gaussian_process.kernels: (ConstantKernel, DotProduct, Matern, RBF, RationalQuadratic, WhiteKernel)

end # module


#=
using Main.MscThesis.ScikitLearnModels
model = Main.MscThesis.ScikitLearnModels.fit!(Main.MscThesis.ScikitLearnModels.DecisionTreeRegressor(), [[1, 2], [3, 4]], [1, 3])
Main.MscThesis.ScikitLearnModels.get_params(model)
Main.MscThesis.ScikitLearnModels.score(model, [[1, 2], [3, 4]], [2, 1])
Main.MscThesis.ScikitLearnModels.predict(model, [[1, 2], [3, 4]])
Main.MscThesis.ScikitLearnModels.decision_path(model, [[1, 2], [3, 4]])

using Main.ScikitLearnModels
model = Main.ScikitLearnModels.fit!(Main.ScikitLearnModels.GaussianProcessRegressor(), [1 2 ; 3 4], [1 1; 3 3])
Main.ScikitLearnModels.get_params(model)
Main.ScikitLearnModels.score(model, [1 2 ; 3 4], [2, 1])
Main.ScikitLearnModels.predict(model, [9 9;])
=#
