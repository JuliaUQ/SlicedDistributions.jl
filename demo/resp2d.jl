using SlicedDistributions
using DelimitedFiles
using Plots

δ = permutedims(readdlm("data_resp_2d.csv", ','))
d = 7

b = 10_000

sn, lh = SlicedNormal(δ, d, b)
