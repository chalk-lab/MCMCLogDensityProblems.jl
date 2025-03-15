using Test

@testset "All Tests" begin
    include("ad_tests.jl")
    include("visualizations.jl")
    include("qa_tests.jl")
end
