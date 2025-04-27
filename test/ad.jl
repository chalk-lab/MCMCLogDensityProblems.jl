using Test, MCMCLogDensityProblems
using MCMCLogDensityProblems: gen_grad, gen_hvp

@testset "AD Tests" begin
    @testset "gen_grad" begin
        for target in [Banana(), Funnel(), HighDimGaussian(10), OneDimGaussianMixtures(), TwoDimGaussianMixtures()]
            d = dim(target)
            for x in [randn(d), randn(d, 100)]
                v, g = logpdf_grad(target, x)
                
                logpdf_grad_ad = gen_grad(_x -> logpdf(target, _x), x)
                v_ad, g_ad = logpdf_grad_ad(x)
                
                @test v ≈ v_ad
                @test g ≈ g_ad
            end
        end
    end
    @testset "gen_hvp" begin
        A = randn(2, 2)
        f = x -> x' * A * x
        x = randn(2)
        v = randn(2)
        
        Hvp_analytical = (A + A') * v
        
        _, _, H = MCMCLogDensityProblems.gen_hess(f, x)(x)
        Hvp = H' * v
        
        Hvp_ad = gen_hvp(f, x, v)(x, v)
        
        @test Hvp_analytical ≈ Hvp ≈ Hvp_ad
    end
end
