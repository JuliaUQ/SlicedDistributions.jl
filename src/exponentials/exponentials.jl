struct SlicedExponential <: SlicedDistribution
    d::Integer
    fp::AbstractFeatureSpace
    λ::AbstractVector
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Float64
end

function SlicedExponential(
    δ::AbstractMatrix,
    d::Integer,
    b::Integer=10000;
    lb::AbstractVector{<:Real}=vec(minimum(δ; dims=2)),
    ub::AbstractVector{<:Real}=vec(maximum(δ; dims=2)),
    basis::Symbol=:poly,
    options::Optim.Options=Optim.Options(; iterations=10^6),
)
    @assert basis in [:poly, :sos]

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], 2d, GradedLexicographicOrder())

    fp = if basis == :poly
        MonomialFeatureSpace(t)
    else
        μ, P = mean_and_precision(t(δ))
        M = cholesky(P).U
        SumOfSquaresFeatureSpace(t, μ, M)
    end

    zδ = fp(δ)
    zΔ = fp(s)

    nz = length(fp)
    V = prod(ub - lb)

    f, ∇f!, ∇²f!, con!, ∇con!, ∇²con! = SlicedDistributions.prepare_optimization(
        zΔ, zδ, V, b
    )

    lx, ux, x0 = if basis == :poly
        fill(-Inf, nz), fill(Inf, nz), zeros(nz)
    else
        fill(0.0, nz), fill(Inf, nz), fill(1e-4, nz)
    end

    @show f(x0)

    # H = zeros(nz, nz)
    # ∇²con!(H, x0, [7.166577e+04])
    # @show H

    # g = zeros(nz)
    # ∇f!(g, x0)
    # H = zeros(nz, nz)
    # ∇²f!(H, x0)

    # ∇f!(g, x0)
    # @show f(x0)
    # @show g
    # @show H

    df = TwiceDifferentiable(f, ∇f!, ∇²f!, x0)
    dfc = TwiceDifferentiableConstraints(con!, ∇con!, ∇²con!, lx, ux, [-Inf], [1e200])

    res = optimize(df, dfc, x0, IPNewton(), options)

    @show res

    cΔ = V / b * sum(exp.(-0.5 * zΔ' * res.minimizer))

    se = SlicedExponential(d, fp, res.minimizer, lb, ub, cΔ)
    return se, -res.minimum
end

function _logpdf(se::SlicedExponential, δ::AbstractArray)
    if all(se.lb .<= δ .<= se.ub)
        return log(exp(-dot(se.fp(δ), se.λ) / 2) / se.c)
    else
        return log(0)
    end
end

function Base.length(se::SlicedExponential)
    return length(se.fp.t[1].x)
end

function Base.eltype(se::SlicedExponential)
    return eltype(se.λ)
end
