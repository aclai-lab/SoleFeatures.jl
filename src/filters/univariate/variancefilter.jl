# ---------------------------------------------------------------------------- #
#                              variance filters                                #
# ---------------------------------------------------------------------------- #
struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # TODO parameters
end

is_unsupervised(::AbstractVarianceFilter) = true

function score(
    X::AbstractDataFrame,
    selector::VarianceFilter
)
    # sum is scaled with n-1
    # var(itr; corrected::Bool=true, mean=nothing[, dims])
    return var.(eachcol(X))
end

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))
# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, ≥))
