using Optim
using ADTypes
using QuasiMonteCarlo
using DelimitedFiles
using SlicedDistributions
using Monomials
using LinearAlgebra
using LogExpFunctions
using Mooncake
using Plots
using Clarabel
using JuMP
using Ipopt
using SCS

δ = readdlm("demo/data/banana.csv", ',')

# Fit Sliced Normal Distribution
d = 4
b = 10000

# lb=[-4, -4]

# ub=[4, 4]

lb = [-3.5, 0]
ub = [3.5, 60]

# lb = [-18, -18]
# ub = [18, 18]

n = size(δ, 2)

const κ = 1.0

s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
fp = SlicedDistributions.MonomialFeatureSpace(t)

zδ = fp(δ)
zΔ = fp(s)

μ, P = SlicedDistributions.mean_and_precision(zδ)

V = prod(ub - lb)

x0 = [μ..., P[tril(trues(size(P)))]...]

f = x -> begin
	μ = x[1:14]
	P = zeros(14, 14)

	for (i, idx) ∈ enumerate(indices)
		P[idx] = x[14+i]
	end

	P = Symmetric(P, :L)

	return -n * (log(V) - log(b) + logsumexp([- SlicedDistributions.ϕ(zΔ[:, i], μ, P)/2.0 for i in 1:b])) - 0.5 * sum([SlicedDistributions.ϕ(zδ[:, i], μ, P) for i in 1:n])
end

f2 = x -> begin
	μ = x[1:14]
	P = zeros(14, 14)

	l = 1
	for i in 1:14
		for j in i:14
			P[i, j] = x[14+l]
			if j != i
				P[j, i] = x[14+l]
			end
			l+=1
		end
	end

	zΔ_μ = zΔ .- μ

	zδ_μ = zδ .- μ

	return @views -n * (log(V) - log(b) + logsumexp([-0.5*(zΔ_μ[:, i]' * P * zΔ_μ[:, i]) for i in 1:b])) - 0.5 * sum([(zδ_μ[:, i]' * P * zδ_μ[:, i]) for i in 1:n])
end

options = Optim.Options(; show_trace = true, iterations = 10^6)
res = optimize(x -> -f2(x), x0, LBFGS(), autodiff = AutoMooncake(), options)

# options = Optim.Options(;show_trace=true)
# res = optimize(x -> -f(x), x0, NelderMead(), options)

μ = Optim.minimizer(res)[1:14]
P = zeros(14, 14)

l = 1
for i in 1:14
	for j in i:14
		P[i, j] = Optim.minimizer(res)[14+l]
		if j != i
			P[j, i] = Optim.minimizer(res)[14+l]
		end
		l+=1
	end
end



Q = vcat(hcat(κ + μ' * P * μ, -μ' * P), hcat(-P * μ, P))

zΔ_μ = zΔ .- μ

# sum(exp.([-0.5*(zΔ_μ[:, i]' * P * zΔ_μ[:, i]) for i in 1:b]))

cΔ = V / b * sum([exp(-SlicedDistributions.ϕ(zΔ[:, i], Q) / 2.0) for i in 1:b])

sn = SlicedNormal(d, fp, Q, lb, ub, 1.0)

xs = range(sn.lb[1], sn.ub[1]; length = 1000)
ys = range(sn.lb[2], sn.ub[2]; length = 1000)

contour(xs, ys, (x, y) -> pdf(sn, [x, y]))

using JuMP

using Ipopt
using HiGHS

V = prod(ub-lb)

using COSMO
# model = Model(Ipopt.Optimizer)
model = Model(Ipopt.Optimizer)
# set_attribute(model, "conic_solver", Clarabel.Optimizer())

set_optimizer_attribute(model, "method", LBFGS())
set_optimizer_attribute(model, "show_trace", true)

n = size(δ, 2)

@variable(model, Q[1:(length(fp)+1), 1:(length(fp)+1)], Symmetric)

# @constraint(model, tr(Q) >= eps())

@variable(model, t)
@variable(model, w[1:b])

@constraint(model, sum(w) <= 1)

# Exponential cone constraints for logsumexp
for i in axes(zΔ, 2)
	zi = zΔ[:, i]
	@constraint(model, [-0.5 * (([1; zi...]' * Q * [1; zi...]) - κ) - t, 1.0, w[i]] in MOI.ExponentialCone())
end

# Q
@expression(model, phi_sum, sum(([1; zδ[:, i]...]' * Q * [1; zδ[:, i]...]) - 1.0 for i ∈ 1:n))
@expression(model, c, V / b * sum(exp(-0.5*(([1; zΔ[:, i]...]' * Q * [1; zΔ[:, i]...]) - 1.0)) for i ∈ 1:b))


# μ, P
@variable(model, μ[1:length(fp)])
@variable(model, P[1:(length(fp)), 1:(length(fp))], Symmetric)
@expression(model, phi_sum, sum(((zδ[:, i] - μ)' * P * (zδ[:, i] - μ)) for i ∈ 1:n))
@expression(model, c, V / b * sum(exp(-0.5*((zΔ[:, i]-μ)' * P * (zΔ[:, i] - μ))) for i ∈ 1:b))

@objective(model, Max, -n * log(c) - 0.5*phi_sum)

@objective(model, Max, -n * (log(V) - log(b) + t) - 0.5*phi_sum)

optimize!(model)

μ = value(mu)

P = value(X)

# Q = vcat(hcat(κ + μ' * P * μ, -μ' * P), hcat(-P * μ, P))

cΔ = V / b * sum([exp(-SlicedDistributions.ϕ(zΔ[:, i], value(Q)) / 2.0) for i in 1:b])

sn = SlicedNormal(d, fp, value(Q), lb, ub, cΔ)

xs = range(sn.lb[1], sn.ub[1]; length = 1000)
ys = range(sn.lb[2], sn.ub[2]; length = 1000)

contour(xs, ys, (x, y) -> pdf(sn, [x, y]))


