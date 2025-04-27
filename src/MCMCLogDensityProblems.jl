module VecTargets

using BSON, Parameters, Distributions, DistributionsAD
using DifferentiationInterface: DifferentiationInterface as DI, AutoReverseDiff
import Distributions: Distribution, dim, rand, logpdf, pdf, VariateForm, Continuous
using Random: AbstractRNG, rand!, randn!, GLOBAL_RNG, shuffle
using ReverseDiff: ReverseDiff
using StatsFuns: logsumexp, logistic

include("bnormals.jl")
include("ad.jl")

include("banana.jl")
export Banana

include("funnel.jl")
export Funnel

include("high_dim_gaussian.jl")
export HighDimGaussian

include("gaussian_mixtures.jl")
export OneDimGaussianMixtures, TwoDimGaussianMixtures

include("spiral.jl")
export Spiral

include("logistic_regression.jl")
export LogisticRegression

include("coxprocess.jl")
export LogGaussianCoxPointProcess

export dim, rand, logpdf, pdf
export logpdf_grad, gen_logpdf_grad, logpdf_hess, gen_logpdf_hess, logpdf_hvp, gen_logpdf_hvp

end # module
