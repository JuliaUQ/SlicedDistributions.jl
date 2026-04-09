using DelimitedFiles
using HCubature
using LinearAlgebra
using Logging
using SlicedDistributions
using StatsBase
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

@testset "Mean and Precision" begin
δ = readdlm("../demo/data/banana.csv", ',')
t = monomials(["δ$i" for i in 1:size(δ, 1)], 10, GradedLexicographicOrder())
zδ = t(δ)
@test !isposdef(cov(zδ;dims=2))
_, P = SlicedDistributions.mean_and_precision(zδ)
@test isposdef(P)
end

include("normals.jl")

@testset "SlicedExponential" begin
    include("exponentials/optim.jl")
    include("exponentials/exponentials.jl")
end
