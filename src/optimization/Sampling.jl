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

"Generates new sample point within the dimensions specified"
function random_sample(bounds::AbstractMatrix)
    checkBounds(bounds)
    # Generate the sample and transpose it to be a column vector
    mapslices(b -> rand() * (b[2] - b[1]) + b[1], bounds, dims=1)'
end

function random_samples(bounds::AbstractMatrix, n::Int, seed::Int=missing)
    !ismissing(seed) ? Random.seed!(seed) : Nothing
    random_samples(bounds, n)
end

function random_samples(bounds::AbstractMatrix, n::Int)
    checkBounds(bounds)
    # Reshape the bounds to be 2 x n
    bounds = size(bounds, 1) == 2 ? bounds : bounds'
    samples = fill(0.0, (size(bounds, 2), n))
    for j in 1:n
        samples[:, j] = random_sample(bounds)
    end
    samples
end



# Tests
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

end
