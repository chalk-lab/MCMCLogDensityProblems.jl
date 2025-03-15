module VecTargets

using BSON, Parameters, Distributions, DistributionsAD
import Distributions: Distributions, VariateForm, ValueSupport, Discrete, Continuous, Distribution, ContinuousMultivariateDistribution, logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, std, var, mode, minimum, maximum
using DifferentiationInterface
using Random: AbstractRNG, rand!, randn!, GLOBAL_RNG, shuffle
using ReverseDiff
using StatsFuns: logsumexp, logistic

const DI = DifferentiationInterface

include("bnormals.jl")
include("ad.jl")

import Distributions: dim, rand, logpdf, pdf

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
