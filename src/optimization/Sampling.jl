module Sampling

using Random

# Macros ---------------------------------------------------------------------
"Generates the `levels` combinations."
macro stratify(levels)
    return :( (Iterators.product($(esc(levels))...)) )
end

# Auxiliar Functions --------------------------------------------------------
"Generate random number between [`min`, `max`)"
rand_range(min, max) = rand() * (max - min) + min

"Sets the seed of the default random pseudogenerator to be `s`"
set_seed!(s::Int) = Random.seed!(seed)

# Public --------------------------------------------------------------------
# All the methods provided in this module generate samples in [0, 1]^n.
# We opt by limiting the samples to the range [0, 1] so that each dimension
# gets scaled according to its properties, i.e., whether it is a discrete, a
# real or a categorical variable, each dimension will be scaled properly.
# The sampling methods should not be responsible for scaling/unscaling the
# variables.

"Generate a new random sample with `ndims` dimensions."
function random_sample(ndims::Int)
    [rand() for j in 1:ndims]
end

"Generate `n` random samples with `ndims` dimensions."
function random_samples(ndims::Int, n::Int)
    samples = zeros(ndims, n)
    for j in 1:n
        samples[:, j] = random_sample(ndims)
    end
    samples
end

"Generate one random sample with `ndims` dimensions for each bin."
function stratifiedMC(ndims, bins)
    if ndims != length(bins)
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

function latinHypercube(ndims, n)
    step = 1 / n
    bin = (i) -> rand_range(step*(i-1), i*step)

    samples = fill(0.0, (ndims, n))
    Sdims = [Set(1:n) for _ in 1:ndims]
    for j in 1:n
        samples[:,j] = map(bin, [pop!(Sdims[d], rand(Sdims[d])) for d in 1:ndims])
    end
    samples
end

function fullfactorial(ndims, level::Int=2)
    if ndims <= 0
        throw(ArgumentError("invalid argument value: ndims $(ndims)")) end
    if level < 2
        throw(ArgumentError("invalid argument value: level $level must be greater than 2")) end
    step = 1 / (level - 1)
    nbins = [0:step:1 for _ in 1:ndims]
    samples = fill(0.0, (ndims, level^ndims))
    for (n, bins) in enumerate(@stratify(nbins))
        samples[:,n] = [b for b in bins]
    end
    samples
end

function boxbehnken(ndims)
    if ndims < 3
        throw(ArgumentError("invalid argument error. ndims $ndims < 3"))
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
             # X[(1+(index-1)*bsize):(index*bsize),1:end .!=i] = X0
         end
    end
    # Append center
    vcat(X, [0.5 for _ in 1:ndims]')'
end


# Tests ---------------------------------------------------------------------
rand_range(0.45, 0.5)

random_sample(0)
random_sample(3)
random_samples(3, 0)
random_samples(3, 1)
random_samples(3, 3)

latinHypercube(1, 4)
latinHypercube(2, 4)
latinHypercube(3, 4)

stratifiedMC(3, [1, 2])
stratifiedMC(2, [3, 4])

fullfactorial(3)
fullfactorial(9)

fullfactorial(2)
fullfactorial(2, 2)
fullfactorial(2, 3)
fullfactorial(0, 3)
fullfactorial(1, 3)
fullfactorial(1, 1)

boxbehnken(2)
boxbehnken(3)
boxbehnken(4)
boxbehnken(5)


# References
# [1] - Giunta, A. A., Wojtkiewicz, S., & Eldred, M. S. (2003).
# Overview of modern design of experiments methods for computational
# simulations. Aiaa, 649(July 2014), 6–9.
# [2] - Box Behnken based on PyDOE2 implementation

end
