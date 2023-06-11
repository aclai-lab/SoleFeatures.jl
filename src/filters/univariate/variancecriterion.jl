struct VarianceCriterion <: AbstractScalarCriterion end

# ========================================================================================
# TRAITS

issupervised(::VarianceCriterion) = false
isunivariate(::VarianceCriterion) = true

# ========================================================================================
# SCORE

function scores(c::VarianceCriterion, X::AbstractDataFrame)::Vector{<:Real}
    # sum is scaled with n-1
    return StatsBase.var.(eachcol(X))
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))
