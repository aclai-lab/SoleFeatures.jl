struct VarianceFilter{T <: AbstractFilterLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # parameters
end

limiter(selector::VarianceFilter) = selector.limiter

# ========================================================================================
# Constructors

# Ranking

is_univariate(::AbstractVarianceFilter{RankingLimiter}) = true
is_unsupervised(::AbstractVarianceFilter{RankingLimiter}) = true
is_supervised(::AbstractVarianceFilter{RankingLimiter}) = true

VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold

is_multivariate(::AbstractVarianceFilter{ThresholdLimiter}) = true
is_unsupervised(::AbstractVarianceFilter{ThresholdLimiter}) = true

VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))

# ========================================================================================
# Shared apply functions

function apply(X::AbstractDataFrame, selector::VarianceFilter{<:AbstractFilterLimiter})
    !is_unsupervised(selector) && throw(ErrorException("Only supervised selector allowed"))
    vars = StatsBase.var.(Iterators.flatten.(eachcol(X)))
    replace!(vars, NaN => -Inf) # (clear or not to clear?)
    return apply_limiter(vars, limiter(selector))
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::VarianceFilter{<:AbstractLimiter}
)
    !is_supervised(selector) && throw(ErrorException("Only supervised selector allowed"))
    numcol = ncol(X)
    original_vars = StatsBase.var.(eachcol(X))
    gdf = _group_by_class(X, y)
    scores = Vector{<:AbstractFloat}(undef, numcol)
    for colidx in 1:numcol
        colvars = StatsBase.var.(eachrow(gdf[:, colidx]))
        scores[colidx] = StatsBase.mean(abs.(original_vars .- colvars))
        # scores[colidx] = maximum(original_vars .- colvars)
        # scores[colidx] = original_vars - StatsBase.mean(colvars)
        # scores[colidx] = original_vars - minimum(colvars)
    end
    return apply_limiter(scores, limiter(selector))
end
