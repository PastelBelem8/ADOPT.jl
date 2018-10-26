module Platypus

using PyCall

const platypus_raw = PyNULL()

# TODO - Verify whether there is a proper platypus installation


function __init__()
    copy!(platypus_raw, pyimport_conda("pandas", "pandas"))
end

"Maps a python object corresponding to a Platypus class to a Julia type which
wraps that class."
const pre_type_map = []
const type_map = Dict{PyObject, Type}()

PyCall.PyObject(x::PandasWrapped) =
    begin
        println("Called PyObject") # Debug
        x.pyo
    end

# macro pytype(name, pyclass, supertype)
#     new_name = "_$(string(name))"
#     quote
#         struct $(new_name) <: $(supertype)
#             pyo::PyObject
#             $(esc(new_name))(pyo::PyObject) = new(pyo)
#             function $(esc(new_name))(args...; kwargs...)
#                 platypus_method = ($pyclass)()
#                 new(pycall(platypus_method, PyObject, args...; kwargs...))
#             end
#         end
#
#         push!(pre_type_map, ($pyclass, $new_name))
#     end
# end


function show(io::IO, df::PandasWrapped)
    s = df.pyo[:__str__]()
    println(io, s)
end

end # Module
