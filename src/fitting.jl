abstract type AbstractFittingMethod end

struct CovarianceScaling <: AbstractFittingMethod
	optimizer::JuMP.MOI.AbstractOptimizer
end
