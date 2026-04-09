@testset "SlicedNormal" begin
    @testset "Banana" begin
        δ = readdlm("../demo/data/banana.csv", ',')

        lb = [-4.0, 0.0]
        ub = [4.0, 65.0]

        d = 2
        b = 40000

        sn, _ = SlicedNormal(δ, d, b; lb, ub)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedNormal(nδ=2, d=2, nz=5,\n  lb=[-4.0, 0.0],\n  ub=[4.0, 65.0])"
    end

    @testset "Circle" begin
        δ = readdlm("../demo/data/circle.csv", ',')

        d = 3
        b = 10000

        lb = [-4.0, -4.0]
        ub = [4.0, 4.0]

        sn, _ = SlicedNormal(δ, d, b; lb, ub)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedNormal(nδ=2, d=3, nz=9,\n  lb=[-4.0, -4.0],\n  ub=[4.0, 4.0])"

        sn, _ = SlicedNormal(δ, d, b)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedNormal(nδ=2, d=3, nz=9,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
    end

    @testset "Swirl" begin
        δ = readdlm("../demo/data/swirl.csv", ',')

        lb = [-18, -18]
        ub = [18, 18]

        d = 7
        b = 80000

        sn, _ = SlicedNormal(δ, d, b; lb, ub)

        shared_tests(sn, δ)

        @test repr(sn) == "SlicedNormal(nδ=2, d=7, nz=35,\n  lb=[-18, -18],\n  ub=[18, 18])"
    end

    @testset "Van-der-Pol" begin
        δ = readdlm("../demo/data/vanderpol.csv", ',')

        lb = [-2.5, -3]
        ub = [2.5, 3.5]

        d = 8
        b = 20000

        sn, _ = SlicedNormal(δ, d, b; lb, ub)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedNormal(nδ=2, d=8, nz=44,\n  lb=[-2.5, -3.0],\n  ub=[2.5, 3.5])"
    end
end
