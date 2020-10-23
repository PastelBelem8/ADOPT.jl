# Unit tests to module utils

@testset "Matrix functions" begin
    A = [1 2 3 4; 5 6 7 8]
    B = Array{Float64}(undef, 1, 0)
    C = Array{Float64}(undef, 0, 5)
    @testset "Rows" begin
        @test utils.nrows(A) = 2
        @test utils.nrows(B) = 1
        @test utils.nrows(C) = 0
    end
    @testset "Cols" begin
        @test utils.ncols(A) = 4
        @test utils.nrows(B) = 0
        @test utils.nrows(C) = 5
    end
end
