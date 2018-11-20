# module utils

using Dates

# Matrix  -------------------------------------------------------------------

export nrows,
       ncols

@inline nrows(A::AbstractMatrix) = size(A, 1)
@inline ncols(A::AbstractMatrix) = size(A, 2)

# Scaling -------------------------------------------------------------------

export  featureScale,
        minMaxScale,
        unitScale
"""
    unitScale(a0[, min, max]) -> a1
    featureScale(a0[, min, max]) -> a1

Scales each value of the column vectors in `a0` to the interval [0, 1].
If `a0` is a type of [`AbstractMatrix`](@ref) the scaling will occur row-wise.
If `min` and `max` are specified, the scaling will consider these values for the
specified dimensions.

See also: [`minMaxScale`](@ref)
"""
unitScale(a::AbstractVector, min::Number, max::Number) = (a .- min) ./ (max .- min)
unitScale(A::AbstractMatrix, min::Number, max::Number) =
    mapslices(a -> unitScale(a, min, max), A, dims=dim)
unitScale(A::AbstractMatrix, min::AbstractVector, max::AbstractVector) = begin
    if size(A, 1) != length(min) || length(min) != length(max)
        throw(DimensionMismatch("number of rows in A $(size(A,1)) should be
        equal to length both vectors min and max: $min $max, respectively."))
    end

    sA = copy(A')
    for j in 1:length(min)
        sA[:, j] = unitScale(A[j,:], min[j], max[j])
    end
    sA'
end
unitScale(a::AbstractVector) = unitScale(a, minimum(a), maximum(a))
unitScale(A::AbstractMatrix) = mapslices(unitScale, A, dims=2)

featureScale = unitScale

"""
    minMaxScale(nmin, nmax, a0) -> a1

Scales each value of the column vectors in `a0` to the interval [`nmin`, `nmax`].
If `a0` is a type of AbstractMatrix the scaling will occur row-wise.

While there are no validations enforcing `nmin` to be lesser or equal to `nmax`,
be aware that specifying a value of `nmin` greater than the value of `nmax` will
invert the magnitude of the vectors to be scaled.

See also: [`featureScale`](@ref), [`unitScale`](@ref)

# Examples
```julia-repl
julia>  a, nmin, nmax = [9, 5, 10, 7], 0, 5;

julia> minMaxScale(nmin, nmax, a)
4-element Array{Float64,1}:
 4.0
 0.0
 5.0
 2.0

julia> minMaxScale(nmax, nmin, a)
4-element Array{Float64,1}:
 1.0
 5.0
 0.0
 3.0

julia> A = rand(50:100, 3, 4)
3×4 Array{Int64,2}:
 59  75  83  74
 99  94  77  51
 53  84  72  53

julia> minMaxScale(nmin, nmax, A) # Row-wise scaling
3×4 Array{Float64,2}:
 0.0  3.33333  5.0      3.125
 5.0  4.47917  2.70833  0.0
 0.0  5.0      3.06452  0.0

```
"""
minMaxScale(nmin::Number, nmax::Number, a::AbstractVector) =
    a |> unitScale |> x -> x * (nmax-nmin) .+ nmin
minMaxScale(nmin::Number, nmax::Number, A::AbstractMatrix, dim::Int) =
    mapslices(a -> min_max_scale(nmin, nmax, a), A, dims=dim)

unscale(value, nmin, nmax, omin=0, omax=1) =
    (value - omin) / (omax - omin) * (nmax - nmin) - nmin

unscale(values::AbstractArray, nmins::AbstractVector, nmaxs::AbstractVector,
        omins::AbstractVector, omaxs::AbstractVector) =
    map(1:size(values, 2)) do j
        unscale(values[:,j], nmins[j], nmaxs[j], omins[j], omaxs[j])
    end

# --------------------------------------------------------------------------
# File
# --------------------------------------------------------------------------
export  create_temporary_file,
        withOutputFile

function withOutputFile(filename::String, do_f::Function)
    @info "[$(now())] Opening file $filename in writing mode..."
    open(filename, "a") do f
        do_f(f)
    end
    @info "[$(now())] Closing file $filename..."
end

create_temporary_file(dir::String, ext::String) =
    joinpath(dir, "$(Dates.format(Dates.now(), "yyyymmddHHMMSS"))") * ext


# CSV utils
csv_sep = ','
"Changes the language of the CSV files to be written"
function csv_language end
function csv_language(lang::Symbol=:EN)
    global csv_sep = lang == :PT ? ";" : ","
    @debug "[$(now())] Changed CSV language to $lang"
end
csv_language(lang::String="EN") = Symbol(lang) |> csv_language

global csv_filename = "results.csv"
csv_file(filename)  = global csv_filename = filename

"Writes a set of `values` in a CSV format in file `filename`"
function csv_write(values, mode="a", el="\n")
    @debug "[$(now())] Writing to file $csv_filename the values:\n$(values)"
    open(csv_filename, mode) do f
        join(f, values, csv_sep)
        write(f, el)
    end
end

function csv_read(filename::String)
    @debug "[$(now())] Reading file $filename"
    content = ""
    open(filename, "r") do f
        content = read(f, String)
    end

    rows = split(rstrip(content), "\n")
    # Return cells
    map(row-> split(row, csv_sep), rows)
end



# --------------------------------------------------------------------------
# Command Line
# --------------------------------------------------------------------------
export  makeWSLcompatible,
        runWSL


function runWSL(executable, args...)
    @debug "Running WSL command. Using file $args."
    args = join([makeWSLcompatible(arg) for arg in args], " ", " ")
    res = chomp(read(`wsl ./$executable $args`, String))
    res = parse(Float64, res)
end

makeWSLcompatible(filepath) =
    replace(filepath, "\\" => "/") |> x -> replace(x, r"(\w+)?:" => lowercase) |> x ->
    replace(x, r"(\w+)?:" => s"/mnt/\1")

# end # Module
