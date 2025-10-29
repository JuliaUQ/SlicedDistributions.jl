abstract type AbstractFeatureSpace end

struct MonomialFeatureSpace <:AbstractFeatureSpace
    t::Vector{Monomial}
end

Base.length(fp::MonomialFeatureSpace) = length(fp.t)

function (fp::MonomialFeatureSpace)(x::AbstractVecOrMat)
    return fp.t(x)
end

struct SumOfSquaresFeatureSpace <:AbstractFeatureSpace
    t::Vector{Monomial}
    μ::Vector{<:Real}
    M::Matrix{<:Real}
end

Base.length(fp::SumOfSquaresFeatureSpace) = length(fp.t)

function (fp::SumOfSquaresFeatureSpace)(x::AbstractVecOrMat)
    z  = fp.t(x)
    return (fp.M * (z .- fp.μ)) .^ 2
end
