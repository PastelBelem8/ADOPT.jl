module Sampling

using DelimitedFiles
using Random

# Macros ---------------------------------------------------------------------
"Generates the `levels` combinations."
macro stratify(levels)
    return :( (Iterators.product($(esc(levels))...)) )
end

# Auxiliar Functions --------------------------------------------------------
"Generate random number between [`minimm`, `maximm`)"
rand_range(minimm::Real, maximm::Real) = rand() * (maximm - minimm) + minimm

"Sets the seed of the default random pseudogenerator to be `s`"
set_seed!(s::Int) = Random.seed!(s)

# Public --------------------------------------------------------------------
# All the methods provided in this module generate samples in [0, 1]^n.
# We opt by limiting the samples to the range [0, 1] so that each dimension
# gets scaled according to its properties, i.e., whether it is a discrete, a
# real or a categorical variable, each dimension will be scaled properly.
# The sampling methods should not be responsible for scaling/unscaling the
# variables.

export  randomMC,
        stratifiedMC,
        latinHypercube,
        kfactorial,
        boxbehnken

"Generate a new random sample with `ndims` dimensions."
function random_sample(ndims::Int)
    if ndims < 0
        throw(DomainError("number of dimensions must be positive")) end
    [rand() for j in 1:ndims]
end

"Generate `n` random samples with `ndims` dimensions."
function randomMC(ndims::Int, n::Int)
    if ndims < 0 || n < 0 throw(DomainError("ndims and n must be positive integers")) end
    samples = zeros(ndims, n)
    for j in 1:n
        samples[:, j] = random_sample(ndims)
    end
    samples
end

"Generate one random sample with `ndims` dimensions for each bin."
function stratifiedMC(ndims::Int, bins::Vector{Int})
    if ndims < 0 || any(map(x -> x < 1, bins))
        throw(DomainError("ndims and bins must be positive integers"))
    elseif ndims != length(bins)
        throw(DimensionMismatch("ndims $(ndims) != bins $(length(bins))"))
    end

    steps = 1 ./ bins
    bin = (b, step) -> rand_range(step*(b-1), b*step)

    samples = fill(0.0, (ndims, prod(bins)))
    for (n, bs) in enumerate(@stratify(map(n-> 1:n, bins)))
        samples[:, n] = [bin(b, steps[i]) for (i, b) in enumerate(bs)]
    end
    samples
end

function latinHypercube(ndims::Int, nbins::Int)
    if ndims < 0 || nbins < 0 throw(DomainError("ndims and n must be positive integers")) end

    step = 1 / nbins
    bin(i) = rand_range(step*(i-1), i*step)
    samples = zeros(ndims, nbins)
    set_of_dims = [Set(1:nbins) for _ in 1:ndims]

    for j in 1:nbins
        samples[:,j] = map(bin, [pop!(set_of_dims[d], rand(collect(set_of_dims[d]))) for d in 1:ndims])
    end
    samples
end

function kfactorial(ndims::Int, level::Int=2)
    if ndims < 0
        throw(DomainError("invalid argument value: ndims $(ndims)")) end
    if level < 2
        throw(DomainError("invalid argument value: level $level must be greater than 2")) end
    step = 1 / (level - 1)
    nbins = [0:step:1 for _ in 1:ndims]
    samples = fill(0.0, (ndims, level^ndims))
    for (n, bins) in enumerate(@stratify(nbins))
        samples[:,n] = [b for b in bins]
    end
    samples
end

function boxbehnken(ndims::Int)
    if ndims < 3
        throw(DomainError("invalid argument error. ndims $ndims < 3"))
    end
    # Block parameters
    X0 = kfactorial(2, 2)'
    bsize = size(X0, 1)
    nblines = convert(Int, 0.5*ndims*(ndims-1)) * bsize
    # Create the samples for each combination
    index = 0
    X = zeros(nblines, ndims)
    for i in 1:(ndims-1)
        for j in (i+1):ndims
            index = index + 1
            X[(1+(index-1)*bsize):(index*bsize),1:end] .= 0.5
            X[(1+(index-1)*bsize):(index*bsize),[i, j]] = X0
        end
    end
    # Append center
    vcat(X, [0.5 for _ in 1:ndims]')'
end

get_existing(sampling_f; kwargs...) =
    if sampling_f == randomMC
        (ndims, nsamples) -> random_sample(ndims, nsamples)
    elseif sampling_f == stratifiedMC
        (ndims, _) -> stratifiedMC(ndims, kwargs[:bins])
    elseif sampling_f == latinHypercube
        (ndims, _) -> latinHypercube(ndims, kwargs[:nbins])
    elseif sampling_f == kfactorial
        (ndims, _) -> kfactorial(ndims, kwargs[:level])
    elseif sampling_f == boxbehnken
        (ndims, _) -> boxbehnken(ndims)
    else
        throw(error("unimplemented sampling function"))
    end

exists(f) = f ∈ (boxbehnken, kfactorial, latinHypercube, randomMC, stratifiedMC)

# ------------------------------------------------------------------------
# Creation Routines
# ------------------------------------------------------------------------
"""
There are two main creation approaches: (1) Sampling-based and (2) File-based.
The first one consists on using a sampling method to create multiple
configurations for the parameters and then evaluate each sample using an
`evaluator`. Note that it is also possible to read the samples from a file,
instead of invoking a sampling method.

The second approach consists in loading the whole information from a file, i.e.,
it is not necessary to generate new samples, nor to evaluate them because that
information is already provided out-of-the-box in the specified file.

# Examples
julia>

"""
generate_samples(;  nvars, nsamples, sampling_function, evaluate, unscalers=[],
                    clip::Bool=false, transform=identity, _...) = let
        unscale(V) = for (index, unscaler) in enumerate(unscalers);
                        V[index,:] = unscaler(V[index,:]); nothing
                    end
        clip_it(val, limit) = clip ? min(val, limit) : val

        X = sampling_function(nvars, nsamples); unscale(X);
        X = X[:, 1:clip_it(size(X, 2), nsamples)]
        X = transform(X)
        y = [evaluate(X[:, j]) for j in 1:size(X, 2)]
        y = cat(y..., dims=2) # concatenate column-wise
        X, y
    end

store_samples(;filename, header=nothing, dlm=',', gensamples_kwargs...) =
    let (X, y) = generate_samples(;gensamples_kwargs...)
        open(filename, "w") do io
            if header != nothing
                join(io, header, dlm)
                write(io, '\n')
            end
            writedlm(io, vcat(X, y)', dlm)
        end
        X, y
    end

load_samples(;nsamples=(:), vars_cols, objs_cols, filename, has_header::Bool=true, dlm=',', _...) =
    let data = open(filename, "r") do io
                    has_header ? readline(io) : nothing;
                    readdlm(io, dlm, Float64, '\n')
                end
        X, y = data[nsamples, vars_cols], data[nsamples, objs_cols]
        X', y'
    end

"""
    create_samples(; kwargs...)

Dispatches the sampling routines according to the provided arguments.
There are three main sampling routines:
- [`load_samples`](@ref): given a `filename`, loads the samples from the file.
It is necessary to know which columns refer to the variables and which columns
refer to the objectives, and therefore it requires the `vars_cols` and `objs_cols`
to be specified. If the argument `nsamples` is supplied, then it will return
the first `nsamples` that were loaded from the file `filename`, otherwise it
will return all the samples.

- [`generate_samples`](@ref): given a `sampling_function`, the number of
dimensions `nvars`, and the number of samples `nsamples`, applies the sampling
function to the `nvars` and `nsamples` parameters and obtains a set of samples.
Since not all sampling routines depend on both parameters, if `clipped` is
specified, the number of samples will be clipped, i.e., it will return at most
the specified nsamples. If `clipped` is not specified, then the result of
applying the sampling function will be returned. If `unscalers` are specified
they will unscale the result of the sampling routines. The unscalers should be
an array of unscaling functions receiving a new value, the current minimum and
the current maximum per dimension. It is assumed that the unscaling functions
already possess the knowledge of the variables bounds within the function as
free variables.

- [`store_samples`](@ref): given a `sampling_function` and a `filename` it
first generate samples using the [`generate_samples`](@ref) method and then
stores it in the specified `filename`.

# Arguments
- `nvars::Int`: the number of variables, i.e., dimensions of the samples.
- `nsamples::Int`: the number of samples to be created.
- `sampling::Function`: the sampling function to be used. It must be a function
    receiving two parameters: the number of variables and the number of samples
- `evaluate::Function`: the function that will receive a sample and produce the
    corresponding objective value.
- `unscalers::Vector{Function}=[]`: an array with the unscalers for each variable.
The provided sampling algorithms are unitary, producing samples with ranges in
    the interval [0, 1]. Each unscaler function must receive a value to be unscaled,
    as well as the old minimum and the old maximum. Defaults to empty vector, in
    which case no unscalers will be applied.
- `clip::Bool=false`: variable indicating if we strictly want the specified `nsamples`.
This is necessary as there are many sampling algorithms for which the number of
    samples is exponential in the number of dimensions (`nvars`).
- `transform::Function=identity`: function that allows to apply a transformation to the
    samples that are produced by the sampling algorithm. It receives as argument
    the matrix with the unscaled samples produced by the sampling function.
- `filename::String`: the filename to read samples from or to store samples to.
    Mandatory when reading samples from file.
- `header::Vector{String}=nothing`: the header to insert in the file when storing the
    samples. If not specified, the file will not have header.
- `has_header::Bool=false`: indicator of whether there exists an header in the file
    from which the samples will be read. Defaults to false.
- `dlm::Char=','`: the delimiter of the sample values in the file.
- `vars_cols::Vector{Int}`: the columns corresponding to the variables that will
    be loaded from the file.
- `objs_cols::Vector{Int}`: the columns corresponding to the objectives that
    will be loaded from the file.

"""
create_samples(;kwargs...) =
    if !haskey(kwargs, :sampling_function)
        haskey(kwargs, :filename) ?
            load_samples(;kwargs...) :
            throw(ArgumentError("invalid sampling methods"))
    else
        λ = kwargs[:sampling_function]
        λ = exists(λ) ? get_existing(λ; kwargs...) : λ
        haskey(kwargs, :filename) ?
            store_samples(; sampling_function=λ, kwargs...) :
            generate_samples(; sampling_function=λ, kwargs...)
    end

export create_samples

# References
# [1] - Giunta, A. A., Wojtkiewicz, S., & Eldred, M. S. (2003).
# Overview of modern design of experiments methods for computational
# simulations. Aiaa, 649(July 2014), 6–9.
# [2] - Box Behnken based on PyDOE2 implementation

end
