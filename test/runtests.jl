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

include("normals.jl")

@testset "SlicedExponential" begin
    include("exponentials/optim.jl")
    include("exponentials/exponentials.jl")
end
