const κ = 1.0

using SlicedDistributions
using DelimitedFiles
using Clarabel
using MosekTools
using SCS
using Plots
using Monomials
using Convex
using QuasiMonteCarlo

δ = permutedims(readdlm("data_resp_2d.csv", ','))
d = 7

b = 10_000

lb = vec(minimum(δ; dims=2))
ub = vec(maximum(δ; dims=2))

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
fp = SlicedDistributions.MonomialFeatureSpace(t)

s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

zδ = fp(δ)
zΔ = fp(s)

μ, P = SlicedDistributions.mean_and_precision(zδ)

V = prod(ub - lb)

m = size(δ, 2)

γ = Variable(1)

problem = minimize(
    m * logsumexp(-γ .* [SlicedDistributions.ϕ(zΔ[:, i], μ, P) / 2.0 for i in 1:b]) + γ * sum([SlicedDistributions.ϕ(zδ[:, i], μ, P) / 2.0 for i in 1:m]),
    γ >= eps(),
)

solver = MOI.OptimizerWithAttributes(SCS.Optimizer, "max_iters" => 10_000_000)

solve!(problem, solver)

P = evaluate(γ) .* P

Q = vcat(hcat(κ + μ' * P * μ, -μ' * P), hcat(-P * μ, P))

cΔ = V / b * sum([exp(-SlicedDistributions.ϕ(zΔ[:, i], Q) / 2.0) for i in 1:b])

sn = SlicedNormal(d, fp, Q, lb, ub, cΔ)

# Plot density
xs = range(sn.lb[1], sn.ub[1]; length=1000)
ys = range(sn.lb[2], sn.ub[2]; length=1000)

contour(xs, ys, (x, y) -> pdf(sn, [x, y]))

samples = rand(sn, 10000)

scatter(samples[1, :], samples[2, :]; label="samples")