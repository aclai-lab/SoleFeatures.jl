"""
Limiter represents a method to select indices of best scores in a scores vector.
"""

abstract type AbstractLimiter end

abstract type AbstractFilterLimiter <: AbstractLimiter end

abstract type AbstractWrapperLimiter <: AbstractLimiter end

abstract type AbstractEmbeddedLimiter <: AbstractLimiter end

"""
    apply_limiter(scores, l)

return indices of best scores
"""
function apply_limiter(scores::AbstractVector{<:Real}, l::AbstractLimiter)
    return error("`apply_limiter` not implemented for type: $(typeof(l))")
end
