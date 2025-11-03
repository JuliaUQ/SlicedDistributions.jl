using DelimitedFiles
using Plots
using SlicedDistributions
using Clarabel

δ = readdlm("demo/data/circle.csv", ',')

# Fit Sliced Normal Distribution
d = 4
b = 10000

# scale the covariance matrix after the initial fit
scaling = CovarianceScaling(Clarabel.Optimizer)

@time sn, lh = SlicedNormal(δ, d, b, [-4, -4], [4, 4]; method = scaling)

samples = rand(sn, 1000)

p = scatter(
	δ[1, :], δ[2, :]; aspect_ratio = :equal, lims = [-4, 4], xlab = "δ1", ylab = "δ2", label = "data",
)
scatter!(p, samples[1, :], samples[2, :]; label = "samples")

display(p)

# Plot density
xs = range(sn.lb[1], sn.ub[1]; length = 1000)
ys = range(sn.lb[2], sn.ub[2]; length = 1000)

contour!(xs, ys, (x, y) -> pdf(sn, [x, y]))
