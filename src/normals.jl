const κ = 1.0

struct SlicedNormal <: SlicedDistribution
    d::Integer
    fp::MonomialFeatureSpace
    μ::AbstractVector{<:Real}
    P::AbstractMatrix{<:Real}
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    c::Real
end

function SlicedNormal(
    δ::AbstractMatrix{<:Real},
    d::Integer,
    b::Integer=10000;
    lb::AbstractVector{<:Real}=vec(minimum(δ; dims=2)),
    ub::AbstractVector{<:Real}=vec(maximum(δ; dims=2)),
)
    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
    fp = MonomialFeatureSpace(t)

    n = size(δ, 2)

    zδ = fp(δ)
    zΔ = fp(s)

    μ, P = mean_and_precision(zδ)

    V = prod(ub - lb)

    P = scale_covariance(μ, P, zδ, zΔ)

    # normalisation constant
    cΔ = V / b * sum([exp(-0.5 * ϕ(zΔ[:, i], μ, P)) for i in 1:b])

    # likelihood
    lh = -n * log(cΔ) - 0.5 * sum([ϕ(zδ[:, i], μ, P) for i in axes(zδ, 2)])

    return SlicedNormal(d, fp, μ, P, lb, ub, cΔ), lh
end

function ϕ(z::Vector{<:Real}, μ::AbstractVector{<:Real}, P::AbstractMatrix{<:Real})
    return (z - μ)' * P * (z - μ)
end

Base.length(sn::SlicedNormal) = length(sn.lb)

function _logpdf(sn::SlicedNormal, δ::AbstractArray)
    if all(sn.lb .<= δ .<= sn.ub)
        z = sn.fp(δ)
        return log(exp(-0.5ϕ(z, sn.μ, sn.P)) / sn.c)
    else
        return log(0)
    end
end

function scale_covariance(
    μ::AbstractVector{<:Real},
    P::AbstractMatrix{<:Real},
    zδ::AbstractMatrix{<:Real},
    zΔ::AbstractMatrix{<:Real},
)
    m = size(zδ, 2)

    sum_ϕ_zδ = sum([ϕ(zδ[:, i], μ, P) / 2.0 for i in axes(zδ, 2)])
    ϕ_zΔ = [ϕ(zΔ[:, i], μ, P) / 2.0 for i in axes(zΔ, 2)]

    function f(γ)
        return m * logsumexp(-γ .* ϕ_zΔ) + γ * sum_ϕ_zδ
    end

    res = optimize(f, eps(), 10000, Brent())

    return Optim.minimizer(res) * P
end
