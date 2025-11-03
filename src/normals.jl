κ = 1.0

struct SlicedNormal <: SlicedDistribution
	d::Integer
	fp::MonomialFeatureSpace
	Q::Matrix{<:Real}
	lb::AbstractVector{<:Real}
	ub::AbstractVector{<:Real}
	c::Real
end

function SlicedNormal(
	δ::AbstractMatrix,
	d::Integer,
	b::Integer = 10000,
	lb::AbstractVector{<:Real} = vec(minimum(δ; dims = 2)),
	ub::AbstractVector{<:Real} = vec(maximum(δ; dims = 2));
	method::Union{Nothing, CovarianceScaling} = nothing)

	s = QuasiMonteCarlo.sample(b, lb, ub, SobolSample())

	t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
	fp = MonomialFeatureSpace(t)

	zδ = fp(δ)
	zΔ = fp(s)

	μ, P = mean_and_precision(zδ)

	# scale covariance matrix before proceeding
	if method isa CovarianceScaling

		m = size(δ, 2)

		ϕΔ = [ϕ(zΔ[:, i], μ, P)/2.0 for i in 1:b]
		ϕδ = [ϕ(zδ[:, i], μ, P)/2.0 for i in 1:m]
		sum_ϕδ = sum(ϕδ)

		model = if method.optimizer isa MOI.AbstractOptimizer
			Model(() -> method.optimizer)
		else
			Model(method.optimizer)
		end

		@variable(model, eps() <= γ <= Inf)
		@variable(model, t)
		@variable(model, w[1:b])

		# Exponential cone constraints for logsumexp
		for i in 1:b
			@constraint(model, [-γ * ϕΔ[i] - t, 1.0, w[i]] in MOI.ExponentialCone())
		end

		@constraint(model, sum(w) <= 1)

		@objective(model, Min, m * t + γ * sum_ϕδ)

		set_silent(model)

		optimize!(model)

		P *= value(γ)
	end

	Q = vcat(
		hcat(κ + μ'*P*μ, -μ'*P),
		hcat(-P*μ, P),
	)

	# normalisation constant
	cΔ = prod(ub-lb) / b * sum([exp(-ϕ(zΔ[:, i], Q)/2.0) for i in 1:b])

	# likelihood in feature space
	lh = sum([log(fz(zδ[:, i], Q) / cΔ) for i in 1:size(δ, 2)])

	return SlicedNormal(d, fp, Q, lb, ub, cΔ), lh
end

function ϕ(z::Vector{<:Real}, Q::AbstractMatrix{<:Real})
	return [1, z...]' * Q * [1, z...] - κ
end

function ϕ(z::Vector{<:Real}, μ::AbstractVector{<:Real}, P::AbstractMatrix{<:Real})
	return (z - μ)'*P*(z-μ)
end

function fz(z::Vector{<:Real}, Q::Matrix{<:Real})
	return exp(-ϕ(z, Q)/2.0)
end

Base.length(sn::SlicedNormal) = length(sn.lb)

function _logpdf(sn::SlicedNormal, δ::AbstractArray)
	if all(sn.lb .<= δ .<= sn.ub)
		z = sn.fp(δ)
		return log(fz(z, sn.Q) / sn.c)
	else
		return log(0)
	end
end
