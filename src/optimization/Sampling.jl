module Sampling

using Random

# To avoid writing boilerplate code, while controlling the thrown exceptions
# macro verifyArg(expr, err)
#     :(if !$(expr) throw($err) end)
# end

# a = [1, 2]
# @macroexpand @verifyArg(ndims(a) == 2, "")

function checkBounds(bounds::AbstractArray)
    if isempty(bounds)
        throw(ArgumentError("empty array is not valid."))
    end
    if ndims(bounds) != 2 || size(bounds, 2) == 0 || size(bounds, 1) != 2
        throw(ArgumentError("dimension mismatch: $(size(bounds)) instead of (2,N) was given."))
    end
end

rand_range(min, max) = rand() * (max - min) + min


"Generates new sample point within the dimensions specified"
function random_sample(bounds::AbstractMatrix)
    checkBounds(bounds)
    # Generate the sample and transpose it to be a column vector
    mapslices(b -> rand_range(b[1], b[2]), bounds, dims=1)'
end

function random_samples(bounds::AbstractMatrix, n::Int)
    checkBounds(bounds)
    # Reshape bounds' dimensions to be 2xn
    bounds = size(bounds, 1) == 2 ? bounds : bounds'
    samples = fill(0.0, (size(bounds, 2), n))
    for j in 1:n
        samples[:, j] = random_sample(bounds)
    end
    samples
end

function random_samples(bounds::AbstractMatrix, n::Int, seed::Int)
    Random.seed!(seed)
    random_samples(bounds, n)
end

macro strats(nbs)
    return :( (Iterators.product((map(n-> 1:n, $(esc(nbs)))...))) )
end

function stratifiedMC(ndims, nbins)
    if ndims != length(nbins)
        throw(ArgumentError("dimension mismatch. ndims $(ndims) != nbins $(length(nbins))")) end
    steps = 1 ./ nbins
    bin = (b, step) -> rand_range(step*(b-1), b*step)

    samples = fill(0.0, (ndims, prod(nbins)))
    for (n, bins) in enumerate(@strats(nbins))
        samples[:, n] = [bin(b, steps[i]) for (i, b) in enumerate(bins)]
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


# Tests

rand_range(0.45, 0.5)


A = [5 9]
random_sample(A)
random_sample(A')
B = [0 0.25; 1 3; 6 25]
random_sample(B)
random_sample(B')
C = [-10 10; -1 1; -3 2]
random_sample(C)
random_sample(C')
D = [-1 -1 -1 -1 -1 -1;
      1  1  1  1  1  1]
random_sample(D)
random_sample(D')
E = Array{Float64}(undef, 0, 0)
random_sample(E)
F = Array{Float64}(undef, 2, 0)
random_sample(F)
G = [-1 -1 -1 -1 -1 -1;
     -1 -1 -1 -1 -1 -1;
      1  1  1  1  1  1]
random_sample(G)
random_samples(D, 3)
random_samples(A', 30)
random_samples(A', 30, 101)

latinHypercube(1, 4)
latinHypercube(2, 4)
latinHypercube(3, 4)

stratifiedMC(3, [1, 2])
stratifiedMC(2, [3, 4])

end

# References
# [1] - Giunta, A. A., Wojtkiewicz, S., & Eldred, M. S. (2003).
# Overview of modern design of experiments methods for computational
# simulations. Aiaa, 649(July 2014), 6–9.
