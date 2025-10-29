κ = 1.0

struct SlicedNormal <: SlicedDistribution
    d::Integer
    fp::MonomialFeatureSpace
    Q::Matrix{<:Real}
    lb::AbstractVector{<:Real}
    ub::AbstractVector{<:Real}
    γ::Real
    c::Real
end

function SlicedNormal(
    δ::AbstractMatrix,
    d::Integer,
    b::Integer=10000,
    lb::AbstractVector{<:Real}=vec(minimum(δ; dims=2)),
    ub::AbstractVector{<:Real}=vec(maximum(δ; dims=2)),
)
    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
    fp = MonomialFeatureSpace(t)

    zδ = fp(δ)
    zΔ = fp(s)

    μ, P = mean_and_precision(zδ)

    Q = vcat(
        hcat(κ + μ'*P*μ, -μ'*P),
        hcat(-P*μ, P)
    )

    # normalisation constants
    γ = (2π)^(length(fp)/2) * sqrt(det(inv(P)))
    cΔ = prod(ub - lb) / b * sum([γ * fz(zΔ[:,i], Q) for i in 1:b])

    # likelihood in feature space
    lh = sum([log(fz(zδ[:, i ], Q)) for i in 1:size(δ,2)])

    return SlicedNormal(d, fp, Q, lb, ub, γ, cΔ), lh
end

function ϕ(z::Vector{<:Real}, Q::Matrix{<:Real})
    return [1, z...]' * Q * [1, z...] - κ
end

function fz(z::Vector{<:Real}, Q::Matrix{<:Real})
    return exp(-ϕ(z,Q)/2.0) / sqrt(κ * (2π)^length(z) * det(inv(Q)))
end

Base.length(sn::SlicedNormal) = length(sn.lb)

function _logpdf(sn::SlicedNormal, δ::AbstractArray)
    if all(sn.lb .<= δ .<= sn.ub)
        z = sn.fp(δ)
        return log((sn.γ / sn.c) * fz(z, sn.Q))
    else
        return log(0)
    end
end
