using Test

@testset "All Tests" begin
    include("ad.jl")
    include("visualizations.jl")
    include("qa_tests.jl")
end
