using DelimitedFiles
using Plots
using SlicedDistributions
using QuasiMonteCarlo
using JuMP
using SCS
using Monomials
using Clarabel

δ = readdlm("demo/data/circle.csv", ',')

# Fit Sliced Normal Distribution
d = 4
b = 10000

s = QuasiMonteCarlo.sample(b, [-4, -4], [4, 4], SobolSample())

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
fp = SlicedDistributions.MonomialFeatureSpace(t)

zδ = fp(δ)
zΔ = fp(s)

μ, P = SlicedDistributions.mean_and_precision(zδ)

# @time sn, lh = SlicedNormal(δ, d, b, [-4, -4], [4, 4])

# println("Likelihood: $lh")

n = size(zδ, 2)

function ϕ(z::Vector{<:Real}, μ::Vector{<:Real}, P::AbstractMatrix{<:Real})
	return (z - μ)'*P*(z-μ) / 2.0
end

ub = [4, 4]
lb = [-4, -4]

fc = γ -> begin
	return prod(ub-lb) / b * sum(exp.([-ϕ(zΔ[:, i], μ, γ * P) for i in 1:b]))
end

f = γ -> begin
	c = fc(γ)
	return γ * sum([ϕ(zδ[:, i], μ, P) for i in 1:n]) - b * log(1/c)
end

f2 = γ -> begin
	return n * logsumexp([-γ * ϕ(zΔ[:, i], μ, P) for i in 1:b]) - γ * sum([ϕ(zδ[:, i], μ, P) for i in 1:n])
end

ϕΔ = [ϕ(zΔ[:, i], μ, P)/2.0 for i in 1:b]
ϕδ = [ϕ(zδ[:, i], μ, P)/2.0 for i in 1:n]

sum_ϕδ = sum(ϕδ)

model = Model(Clarabel.Optimizer)
# set_attribute(model, "max_iters", 10^5)
@variable(model, eps() <= γ <= Inf)
@variable(model, t)
@variable(model, w[1:b])

# Exponential cone constraints for logsumexp
for i in 1:b
	@constraint(model, [-γ * ϕΔ[i] - t, 1.0, w[i]] in MOI.ExponentialCone())
end

@constraint(model, sum(w) <= 1)

@objective(model, Min, n * t + γ * sum_ϕδ)

optimize!(model)

# model = Model(Optim.Optimizer)

# @variable(model, γ)

# using LogExpFunctions

# @objective(model, Max, -b * log(sum(exp.(-γ .* ϕΔ))) - γ * sum_ϕδ)

# @objective(-b * logsumexp([-γ * ϕΔ[i] for i in 1:b]) - γ * su))
samples = rand(sn, 1000)

Popt = value(γ) * P

κ = 1.0
Q = vcat(
	hcat(κ + μ'*Popt*μ, -μ'*Popt),
	hcat(-Popt*μ, Popt),
)

c = prod(ub-lb) / b * sum([exp(-ϕ(zΔ[:, i], μ, Popt)) for i in 1:b])


sn = SlicedNormal(d, fp, Q, [-4, -4], [4, 4], c)
p = scatter(
	δ[1, :], δ[2, :]; aspect_ratio = :equal, lims = [-4, 4], xlab = "δ1", ylab = "δ2", label = "data",
)
scatter!(p, samples[1, :], samples[2, :]; label = "samples")

display(p)

# Plot density
xs = range(sn.lb[1], sn.ub[1]; length = 1000)
ys = range(sn.lb[2], sn.ub[2]; length = 1000)

contour!(xs, ys, (x, y) -> pdf(sn, [x, y]))
