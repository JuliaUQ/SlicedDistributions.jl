@testset "Derivatives" begin
    δ = readdlm("../demo/data/circle.csv", ',')
    d = 4
    b = 10000

    lb = [-4.0, -4.0]
    ub = [4.0, 4.0]

    V = prod(ub - lb)

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], 2d, GradedLexicographicOrder())

    fp = SlicedDistributions.MonomialFeatureSpace(t)
    nz = length(fp)

    zδ = fp(δ)
    zΔ = fp(s)

    f, ∇f!, ∇²f!, con!, ∇con!, ∇²con! = SlicedDistributions.prepare_optimization(
        zΔ, zδ, V, b
    )

    λ = zeros(nz)

    @testset "Objective" begin
        g_AD = zeros(nz)
        g_analytical = zeros(nz)

        ForwardDiff.gradient!(g_AD, f, λ)
        ∇f!(g_analytical, λ)

        @test isapprox(g_AD, g_analytical)

        H_AD = zeros(nz, nz)
        H_analytical = zeros(nz, nz)

        ForwardDiff.hessian!(H_AD, f, λ)
        ∇²f!(H_analytical, λ)

        @test isapprox(H_AD, H_analytical)
    end

    @testset "Constraints" begin
        J_AD = zeros(1, nz)
        J_analytical = zeros(1, nz)

        ForwardDiff.jacobian!(J_AD, x -> con!([zero(eltype(x))], x), λ)
        ∇con!(J_analytical, λ)

        @test isapprox(J_AD, J_analytical)

        H_AD = zeros(nz, nz)
        H_analytical = zeros(nz, nz)

        l = [0.5]

        ∇²f!(H_AD, λ)

        H_AD += l[1] * ForwardDiff.hessian(x -> con!([zero(eltype(x))], x)[1], λ)

        ∇²con!(H_analytical, λ, l)

        @test isapprox(H_AD, H_analytical)
    end
end
