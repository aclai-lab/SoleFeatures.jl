struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_unsupervised(::AbstractVarianceFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    selector::VarianceFilter
)
    # sum is scaled with n-1
    return StatsBase.var.(eachcol(X))
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))
