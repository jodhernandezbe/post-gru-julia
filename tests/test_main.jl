# Import libraries
using Test

@testset "Main Test Suite" begin
    include("test_data_preprocessing.jl")
    include("test_scratch_gru.jl")
end