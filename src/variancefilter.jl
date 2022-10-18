struct VarianceFilter{T <: AbstractFilterLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # parameters
end

is_unsupervised(::AbstractVarianceFilter{AbstractFilterLimiter}) = true
is_univariate(::AbstractVarianceFilter{RankingLimiter}) = true
is_multivariate(::AbstractVarianceFilter{ThresholdLimiter}) = true

limiter(selector::VarianceFilter) = selector.limiter

# Ranking constructor

VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold constructor

VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))

# Shared apply

function apply(df::AbstractDataFrame, selector::VarianceFilter{T}) where {T <: AbstractFilterLimiter}
    vars = [ StatsBase.var(Iterators.flatten(c)) for c in eachcol(df) ]
    replace!(vars, NaN => -Inf)
    return apply_limiter(vars, limiter(selector))
end
