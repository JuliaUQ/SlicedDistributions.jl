using DelimitedFiles
using HCubature
using Logging
using SlicedDistributions
using Test
using ForwardDiff
using QuasiMonteCarlo
using Monomials

# Disable TMCMC Output during tests
Logging.disable_logging(Logging.Info)

function shared_tests(sn::SlicedDistributions.SlicedDistribution, δ::AbstractMatrix{<:Real})
    @test all(insupport.(sn, eachcol(δ)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))
    @test all(pdf.(sn, eachcol(samples)) .>= 0)

    @test pdf(sn, sn.lb .- 1) == 0.0
    @test pdf(sn, sn.ub .+ 1) == 0.0
end

include("normals.jl")

@testset "SlicedExponential" begin
    include("exponentials/optim.jl")
    include("exponentials/exponentials.jl")
end
