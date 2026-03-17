@testset "Sum of Squares" begin
    @testset "Circle" begin
        δ = readdlm("../demo/data/circle.csv", ',')

        d = 3
        b = 20000

        lb = [-4.0, -4.0]
        ub = [4.0, 4.0]

        sn, _ = SlicedExponential(δ, d, b; lb, ub, basis=:sos)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-4.0, -4.0],\n  ub=[4.0, 4.0])"

        sn, _ = SlicedExponential(δ, d, b; basis=:sos)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
    end

    @testset "Swirl" begin
        δ = readdlm("../demo/data/swirl.csv", ',')

        lb = [-18, -18]
        ub = [18, 18]

        d = 4
        b = 120000

        sn, _ = SlicedExponential(δ, d, b; lb, ub, basis=:sos)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedExponential(nδ=2, d=4, nz=44,\n  lb=[-18, -18],\n  ub=[18, 18])"
    end

    @testset "Van-der-Pol" begin
        δ = readdlm("../demo/data/vanderpol.csv", ',')

        lb = [-2.5, -3]
        ub = [2.5, 3.5]

        d = 4
        b = 100000

        sn, _ = SlicedExponential(δ, d, b; lb, ub, basis=:sos)

        shared_tests(sn, δ)

        @test repr(sn) ==
            "SlicedExponential(nδ=2, d=4, nz=44,\n  lb=[-2.5, -3.0],\n  ub=[2.5, 3.5])"
    end
end

@testset "Polynomial" begin
    δ = readdlm("../demo/data/circle.csv", ',')

    d = 3
    b = 30000

    sn, _ = SlicedExponential(δ, d, b)

    shared_tests(sn, δ)

    @test repr(sn) ==
        "SlicedExponential(nδ=2, d=3, nz=27,\n  lb=[-3.2929135595124106, -3.453912509293032],\n  ub=[3.3151850678141344, 3.3768332192657207])"
end
