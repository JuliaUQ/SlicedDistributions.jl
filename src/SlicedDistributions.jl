module SlicedDistributions
using Distributions
using LinearAlgebra
using LogExpFunctions
using Monomials
using NearestCorrelationMatrix
using StatsBase
using TransitionalMCMC
using QuasiMonteCarlo
using Optim
using PreallocationTools
using Random

import Base: eltype, length, show
import Distributions: _logpdf, insupport
export SlicedNormal, SlicedExponential

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
    Σ = (cov(z; dims=2))

    if isposdef(Σ)
        return μ, inv(Symmetric(Σ))
    else
        C = Symmetric(cor(z; dims=2))
        nearest_cor!(C)
        s = std(z; dims=2)

        return μ, inv(Symmetric(cor2cov(Matrix(C), s)))
    end
end

include("featurespace.jl")
include("normals.jl")
include("exponentials/optim.jl")
include("exponentials/exponentials.jl")

Base.broadcastable(sd::SlicedDistribution) = Ref(sd)

function Base.show(io::IO, sd::SlicedDistribution)
    print(io, "$(typeof(sd))(nδ=$(length(sd)), d=$(sd.d), nz=$(length(sd.fp)),\n")
    print(io, "  lb=$(sd.lb),\n")
    print(io, "  ub=$(sd.ub))")
    return nothing
end

end
