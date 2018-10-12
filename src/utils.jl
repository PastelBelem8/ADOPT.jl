module utils
export nrows, ncols

@inline nrows(A::AbstractMatrix) = size(A, 1)
@inline ncols(A::AbstractMatrix) = size(A, 2)
end
