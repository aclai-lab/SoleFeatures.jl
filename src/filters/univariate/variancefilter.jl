struct VarianceFilter{T <: AbstractLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_supervised(::AbstractVarianceFilter) = true
is_unsupervised(::AbstractVarianceFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    selector::VarianceFilter
)
    scores = StatsBase.var.(Iterators.flatten.(eachcol(X)))
    replace!(scores, NaN => -Inf) # (clear or not to clear?)
    return scores
end

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::VarianceFilter
)
    numcol = ncol(X)
    original_vars = StatsBase.var.(eachcol(X))
    gdf = _group_by_class(X, y)
    scores = Vector{AbstractFloat}(undef, numcol)
    for colidx in 1:numcol
        splitvars = StatsBase.var.(gdf[:, colidx])
        scores[colidx] = minimum(original_vars[colidx] .- splitvars)
        # scores[colidx] = StatsBase.mean(original_vars[colidx] .- splitvars) # strict version
    end
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))
