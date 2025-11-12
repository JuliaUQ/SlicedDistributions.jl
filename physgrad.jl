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

δ = readdlm("demo/data/circle.csv", ',')

# Fit Sliced Normal Distribution
d = 4
b = 10000

lb=[-4, -4]

ub=[4, 4]

n = size(δ, 2)


s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

t = monomials(["δ$i" for i in 1:size(δ, 1)], d, GradedLexicographicOrder())
fp = SlicedDistributions.MonomialFeatureSpace(t)

zδ = fp(δ)
zΔ = fp(s)

μ, P = SlicedDistributions.mean_and_precision(zδ)

indices = collect(Iterators.filter(c -> c[1] >= c[2], CartesianIndices(P)))

V = prod(ub - lb)

x0 = [μ..., P[tril(trues(size(P)))]...]

f = x -> begin
    μ = x[1:14]
    P = zeros(14,14)

    for (i, idx) = enumerate(indices)
        P[idx] = x[14+i]
    end

    P = Symmetric(P,:L)

    return -n *(log(V) - log(b) + logsumexp([-SlicedDistributions.ϕ(zΔ[:,i], μ, P)/2.0 for i in 1:b])) - 0.5 * sum([SlicedDistributions.ϕ(zδ[:,i], μ, P) for i in 1:n])
end

options = Optim.Options(;show_trace=true, iterations=10)
res = optimize(x -> -f(x), x0, LBFGS(), autodiff=AutoMooncake(), options)

# options = Optim.Options(;show_trace=true)
# res = optimize(x -> -f(x), x0, NelderMead(), options)

μ = Optim.minimizer(res)[1:14]
P = zeros(14, 14)

for (i, idx) = enumerate(indices)
        P[idx] = Optim.minimizer(res)[14+i]
end

P = Symmetric(P, :L)

Q = vcat(hcat(κ + μ' * P * μ, -μ' * P), hcat(-P * μ, P))

cΔ = V / b * sum([exp(-SlicedDistributions.ϕ(zΔ[:, i], Q) / 2.0) for i in 1:b])

sn = SlicedNormal(d, fp, Q, lb, ub, cΔ)

xs = range(sn.lb[1], sn.ub[1]; length = 1000)
ys = range(sn.lb[2], sn.ub[2]; length = 1000)

contour(xs, ys, (x, y) -> pdf(sn, [x, y]))
