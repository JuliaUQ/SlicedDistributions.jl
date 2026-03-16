δ = readdlm("../demo/data/circle.csv", ',')

@testset "Polynomial" begin
    d = 3
    b = 30000

    sn, _ = SlicedExponential(δ, d, b)

    @test all(insupport.(sn, eachcol(δ)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))

    @test repr(sn) ==
        "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
end

@testset "Sum of Squares" begin
    d = 3
    b = 20000

    sn, _ = SlicedExponential(δ, d, b; lb=[-4.0, -4.0], ub=[4.0, 4.0], basis=:sos)

    @test all(insupport.(sn, eachcol(δ)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))

    @test repr(sn) ==
        "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-4.0, -4.0],\n  ub=[4.0, 4.0])"
end
