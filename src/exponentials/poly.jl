struct SlicedExponential <: SlicedDistribution
    d::Integer
    fp::MonomialFeatureSpace
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
    optimizer::Union{Type{<:MOI.AbstractOptimizer}, Nothing} = nothing,
    basis::Symbol=:poly,
)

    @assert basis in [:poly, :sos]

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], 2d, GradedLexicographicOrder())
    fp = MonomialFeatureSpace(t)

    zδ = fp(δ)
    zΔ = fp(s)

    n = size(δ, 2)
    nz = length(fp)
    V = prod(ub-lb)

    model = Model(optimizer)

    @variable(model, λ[1:nz])

    @variable(model, t)
    @variable(model, w[1:b])
    @constraint(model, sum(w) <= 1)

    for i in axes(zΔ, 2)
	zi = zΔ[:, i]
	@constraint(model, [-0.5*dot(zi, λ) - t, 1.0, w[i]] in MOI.ExponentialCone())
    end

    @expression(model, phi_sum, sum(dot(zδ[:, i], λ) for i ∈ 1:n))

    @objective(model, Max, -n * (log(V) - log(b) + t) - 0.5*phi_sum)

    optimize!(model)

    cΔ = V / b * sum([exp(-dot(zΔ[:, i], value(λ)) / 2.0) for i in 1:b])

    se = SlicedExponential(d, fp, value(λ), lb, ub, cΔ)
    return se, objective_value(model)
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
