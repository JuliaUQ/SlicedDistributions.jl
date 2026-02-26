module SlicedDistributions
using Distributions
using JuMP
using LinearAlgebra
using LogExpFunctions
using Monomials
using TransitionalMCMC
using QuasiMonteCarlo
using Optim
using Random

import Base: eltype, length, show
import Distributions: _logpdf, insupport
export SlicedNormal, SlicedExponential
export CovarianceScaling

export pdf, insupport

abstract type SlicedDistribution <: ContinuousMultivariateDistribution end

function Distributions.rand!(rng::AbstractRNG, sd::SlicedDistribution, x::AbstractMatrix)
    prior = Uniform.(sd.lb, sd.ub)

    logprior(x) = sum(logpdf.(prior, x))
    sampler(n) = mapreduce(u -> rand(rng, u, n), hcat, prior)
    loglikelihood(x) = Distributions.logpdf(sd, x)

    samples, _ = tmcmc(loglikelihood, logprior, sampler, size(x, 2))

    x[:] = permutedims(samples)

    return x
end

function Distributions.insupport(sd::SlicedDistribution, x::AbstractVector)
    return all(sd.lb .<= x .<= sd.ub)
end

function mean_and_precision(z::AbstractMatrix)
    μ = vec(mean(z; dims=2))
    P = (inv(Symmetric(cov(z; dims=2))))

    return μ, P
end

include("featurespace.jl")
include("normals.jl")
include("exponentials.jl")

Base.broadcastable(sd::SlicedDistribution) = Ref(sd)

function Base.show(io::IO, sd::SlicedDistribution)
    print(io, "$(typeof(sd))(nδ=$(length(sd)), d=$(sd.d), nz=$(length(sd.fp)),\n")
    print(io, "  lb=$(sd.lb),\n")
    print(io, "  ub=$(sd.ub))")
    return nothing
end

end
