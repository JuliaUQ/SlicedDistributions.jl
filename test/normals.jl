@testset "SlicedNormal" begin
    δ = readdlm("../demo/data/circle.csv", ',')

    d = 3
    b = 10000

    sn, _ = SlicedNormal(δ, d, b)

    @test all(insupport.(sn, eachcol(δ)))

    @test hcubature(x -> pdf(sn, x), sn.lb, sn.ub)[1] ≈ 1.0 atol = 1e-3

    samples = rand(sn, 1000)

    @test all(insupport.(sn, eachcol(samples)))

    @test repr(sn) ==
        "SlicedNormal(nδ=2, d=3, nz=9,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
end
