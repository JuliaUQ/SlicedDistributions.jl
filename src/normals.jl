const κ = 1.0

struct SlicedNormal <: SlicedDistribution
	d::Integer
	fp::MonomialFeatureSpace
	Q::AbstractMatrix{<:Real}
	lb::AbstractVector{<:Real}
	ub::AbstractVector{<:Real}
	c::Real
end

function SlicedNormal(
	δ::AbstractMatrix{<:Real},
	d::Integer,
	b::Integer = 10000;
	lb::AbstractVector{<:Real} = vec(minimum(δ; dims = 2)),
	ub::AbstractVector{<:Real} = vec(maximum(δ; dims = 2)),
	optimizer::Union{Type{<:MOI.AbstractOptimizer}, Nothing} = nothing,
	optimality::Symbol = :feature,
)
	@assert optimality in [:feature, :physical]

	s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

	t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
	fp = MonomialFeatureSpace(t)

	zδ = fp(δ)
	zΔ = fp(s)

	μ, P = mean_and_precision(zδ)

	V = prod(ub - lb)

	local Q

	if optimality == :feature
		if optimizer != nothing
			P = scale_covariance(μ, P, zδ, zΔ, optimizer)
		end
		Q = vcat(hcat(κ + μ' * P * μ, -μ' * P), hcat(-P * μ, P))
	elseif optimality == :physical
		Q = optimality_in_physical_space(μ, P, zδ, zΔ, V, optimizer)
	end

	# normalisation constant
	cΔ = V / b * sum([exp(-ϕ(zΔ[:, i], Q) / 2.0) for i in 1:b])

	# likelihood
	lh = sum([log(fz(zδ[:, i], Q) / cΔ) for i in 1:size(δ, 2)])

	return SlicedNormal(d, fp, Q, lb, ub, cΔ), lh
end

function ϕ(z::Vector{<:Real}, Q::AbstractMatrix{<:Real})
	return [1, z...]' * Q * [1, z...] - κ
end

function ϕ(z::Vector{<:Real}, μ::AbstractVector{<:Real}, P::AbstractMatrix{<:Real})
	return (z - μ)' * P * (z - μ)
end

function fz(z::Vector{<:Real}, Q::AbstractMatrix{<:Real})
	return exp(-ϕ(z, Q) / 2.0)
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

function scale_covariance(
	μ::AbstractVector{<:Real},
	P::AbstractMatrix{<:Real},
	zδ::AbstractMatrix{<:Real},
	zΔ::AbstractMatrix{<:Real},
	optimizer::Type{<:MOI.AbstractOptimizer},
)

	model = Model(optimizer)

	@variable(model, eps() <= γ <= Inf)
	@variable(model, t)
	@variable(model, w[1:size(zΔ, 2)])

	# Exponential cone constraints for logsumexp
	for i in axes(zΔ, 2)
		@constraint(model, [-γ * ϕ(zΔ[:, i], μ, P)/2.0 - t, 1.0, w[i]] in MOI.ExponentialCone())
	end

	@constraint(model, sum(w) <= 1)

	m = size(zδ, 2)
	sumϕδ = sum([ϕ(zδ[:, i], μ, P) / 2.0 for i in axes(zδ, 2)])

	@objective(model, Min, m * t + γ * sumϕδ)

	optimize!(model)

	return value(γ) * P
end

function optimality_in_physical_space(
	μ::AbstractVector{<:Real},
	P::AbstractMatrix{<:Real},
	zδ::AbstractMatrix{<:Real},
	zΔ::AbstractMatrix{<:Real},
	V::Real,
	optimizer::Type{<:MOI.AbstractOptimizer},
)
	m = size(zδ, 2)
	n = length(μ) + 1
	b = size(zΔ, 2)
	model = Model(optimizer)

	@variable(model, Q[1:n, 1:n], PSD)
	@variable(model, t)
	@variable(model, w[1:b])

	# Exponential cone constraints for logsumexp
	for i in axes(zΔ, 2)
		zi = zΔ[:, i]
		@constraint(model, [-(0.5) * ([1; zi]' * Q * [1; zi] - κ) - t, 1.0, w[i]] in MOI.ExponentialCone())
	end

	@constraint(model, sum(w) <= 1)

	@expression(model, phi_sum, sum(([1; zδ[:, i]]' * Q * [1; zδ[:, i]] - κ) for i in 1:m))

	# Objective: maximize
	@objective(model, Max, -m*(log(V) - log(b) + t) - 0.5 * phi_sum)

	optimize!(model)

	return value(Q)
end
