module Sampling

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
        fullfactorial,
        boxbehnken

"Generate a new random sample with `ndims` dimensions."
function random_sample(ndims::Int)
    if ndims < 0
        throw(DomainError("number of dimensions must be positive")) end
    [rand() for j in 1:ndims]
end

"Generate `n` random samples with `ndims` dimensions."
function random_samples(ndims::Int, n::Int)
    if ndims < 0 || n < 0 throw(DomainError("ndims and n must be positive integers")) end
    samples = zeros(ndims, n)
    for j in 1:n
        samples[:, j] = random_sample(ndims)
    end
    samples
end
randomMC = random_samples

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

function fullfactorial(ndims::Int, level::Int=2)
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
    X0 = fullfactorial(2, 2)'
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

# References
# [1] - Giunta, A. A., Wojtkiewicz, S., & Eldred, M. S. (2003).
# Overview of modern design of experiments methods for computational
# simulations. Aiaa, 649(July 2014), 6–9.
# [2] - Box Behnken based on PyDOE2 implementation

end
