abstract type AbstractFittingMethod end

struct CovarianceScaling <: AbstractFittingMethod
	optimizer::Any
end
