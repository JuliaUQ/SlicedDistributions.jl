struct SlicedExponential <: SlicedDistribution
    d::Integer
    fp::AbstractFeatureSpace
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
    optimizer::Union{Type{<:MOI.AbstractOptimizer}}=Optim.Optimizer,
    basis::Symbol=:poly,
)
    @assert basis in [:poly, :sos]

    s = QuasiMonteCarlo.sample(b, lb, ub, HaltonSample())

    t = monomials(["δ$i" for i in 1:size(δ, 1)], 2d, GradedLexicographicOrder())

    fp = if basis == :poly
        MonomialFeatureSpace(t)
    else
        μ, P = mean_and_precision(t(δ))
        M = cholesky(P).U
        SumOfSquaresFeatureSpace(t, μ, M)
    end

    zδ = fp(δ)
    zΔ = fp(s)

    n = size(δ, 2)
    nz = length(fp)
    V = prod(ub - lb)

    model = Model(optimizer)

    if basis == :poly
        @variable(model, λ[1:nz])
    else
        @variable(model, λ[1:nz] .>= 0.0)
    end

    @expression(model, cΔ, V / b * sum(exp.(zΔ' * λ ./ -2)))

    @objective(model, Min, n * log(cΔ) + sum(zδ' * λ) / 2)

    optimize!(model)

    se = SlicedExponential(d, fp, value(λ), lb, ub, value(cΔ))
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
