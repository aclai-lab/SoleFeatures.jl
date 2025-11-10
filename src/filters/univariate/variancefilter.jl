# ---------------------------------------------------------------------------- #
#                              variance filter                                 #
# ---------------------------------------------------------------------------- #
struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractVarianceFilter) = false
is_unsupervised(::AbstractVarianceFilter) = true

function score(X::AbstractArray, selector::VarianceFilter)::Vector{Float64}
    # sum is scaled with n-1
    # var(itr; corrected::Bool=true, mean=nothing[, dims])
    return var.(eachcol(X))
end

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, ≥))
