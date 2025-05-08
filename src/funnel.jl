Base.@kwdef struct Funnel <: ContinuousMultivariateDistribution 
    dim::Int=2
end

dim(funnel::Funnel) = funnel.dim

function _logpdf(::Funnel, θ::AbstractVecOrMat)
    nu, x = θ[1,:], θ[2,:]

    lp1 = _logpdf_normal_std.(nu, 0, 3)
    lp2 = _logpdf_normal_std.(x, 0, exp.(nu ./ 2))
    return lp1 .+ lp2
end

logpdf(funnel::Funnel, θ::AbstractVector) = only(_logpdf(funnel, θ))

logpdf(funnel::Funnel, θ::AbstractMatrix) = _logpdf(funnel, θ)
