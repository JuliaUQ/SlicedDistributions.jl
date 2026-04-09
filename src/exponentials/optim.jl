function prepare_optimization(
    zΔ::AbstractMatrix{<:Real}, zδ::AbstractMatrix{<:Real}, V::Real, b::Real
)
    n = size(zδ, 2)
    cache = LazyBufferCache()

    # objective
    function f(λ)
        return n * (log(V) - log(b) + logsumexp(-0.5 * zΔ' * λ)) + 0.5 * sum(zδ' * λ)
    end
    # objective gradient
    function ∇f!(g, λ)
        expzΔλ = cache[@view zΔ[1, :]]
        expzΔλ .= exp.(-0.5 * zΔ' * λ)
        g .= n * (-0.5 * zΔ * expzΔλ) / sum(expzΔλ) + 0.5 * sum(zδ; dims=2)[:]
        return nothing
    end

    # objective hessian
    function ∇²f!(H, λ)
        expzΔλ = cache[@view zΔ[1, :]]
        expzΔλ .= exp.(-0.5 * zΔ' * λ)
        sum_exp_zΔ = sum(expzΔλ)
        zΔexpzΔλ = -0.5 * zΔ * expzΔλ

        H .=
            n * (zΔ * Diagonal(0.25 .* expzΔλ) * zΔ' .* sum_exp_zΔ - zΔexpzΔλ * zΔexpzΔλ') /
            sum_exp_zΔ^2
        return nothing
    end

    # constraint
    function con!(c, λ)
        c[1] = V / b * sum(exp.(-0.5 * zΔ' * λ))
        return c
    end

    # constraint jacobian
    function ∇con!(J, λ)
        J[1, :] .= V / b * -0.5 * zΔ * exp.(-0.5 * zΔ' * λ)
        return J
    end

    # constraint hessian
    function ∇²con!(H, λ, l)
        ∇²f!(H, λ)
        expzΔλ = cache[@view zΔ[1, :]]
        expzΔλ .= exp.(-0.5 * zΔ' * λ)
        return H .+= l[1] * V / b * zΔ * Diagonal(0.25 .* expzΔλ) * zΔ'
    end

    return f, ∇f!, ∇²f!, con!, ∇con!, ∇²con!
end
