# ---------------------------------------------------------------------------- #
#                              variance filters                                #
# ---------------------------------------------------------------------------- #
struct VarianceFilter{T<:AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractVarianceFilter) = false
is_unsupervised(::AbstractVarianceFilter) = true

function score(X::AbstractMatrix, selector::VarianceFilter)
    # sum is scaled with n-1
    # var(itr; corrected::Bool=true, mean=nothing[, dims])
    return var.(eachcol(X))
end
score(Xdf::AbstractDataFrame, selector::VarianceFilter) = score(Matrix(Xdf), selector)

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))
# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, ≥))
