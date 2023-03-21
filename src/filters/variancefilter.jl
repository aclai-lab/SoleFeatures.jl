struct VarianceFilter{T <: AbstractFilterLimiter} <: AbstractVarianceFilter{T}
    limiter::T
    # parameters
end

is_univariate(::AbstractVarianceFilter) = true
is_supervised(::AbstractVarianceFilter) = true
is_unsupervised(::AbstractVarianceFilter) = true

# ========================================================================================
# Constructors

# Ranking
VarianceRanking(nbest) = VarianceFilter(RankingLimiter(nbest, true))

# Threshold
VarianceThreshold(threshold) = VarianceFilter(ThresholdLimiter(threshold, >=))

# ========================================================================================
# Shared apply functions

function apply(
    X::AbstractDataFrame,
    selector::VarianceFilter{<:AbstractFilterLimiter};
    returnscores=false
)
    !is_unsupervised(selector) && throw(ErrorException("Only unsupervised selector allowed"))
    vars = StatsBase.var.(Iterators.flatten.(eachcol(X)))
    replace!(vars, NaN => -Inf) # (clear or not to clear?)
    if (returnscores) return apply_limiter(vars, limiter(selector)), vars end
    return apply_limiter(vars, limiter(selector))
end

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::VarianceFilter{<:AbstractLimiter};
    returnscores=false
)
    !is_supervised(selector) && throw(ErrorException("Only supervised selector allowed"))
    numcol = ncol(X)
    original_vars = StatsBase.var.(eachcol(X))
    gdf = _group_by_class(X, y)
    scores = Vector{AbstractFloat}(undef, numcol)
    for colidx in 1:numcol
        splitvars = StatsBase.var.(gdf[:, colidx])

        # TODO: start remove
        if ( -1 <= original_vars[colidx] >= 1.5 )
            throw(ErrorException("Original vars not normalized"))
        end
        if ( -1 <= gdf[1,1][1] >= 1.5 )
            throw(ErrorException("Grouped vars not normaized"))
        end
        # println(gdf[:, colidx])
        # println(original_vars)
        # println("====")
        # println(splitvars)
        # println(length(splitvars))
        # println("====")
        # TODO: end remove

        scores[colidx] = minimum(original_vars[colidx] .- splitvars)
        # scores[colidx] = StatsBase.mean(original_vars - colvars) # strict version
    end
    if (returnscores) return apply_limiter(scores, limiter(selector)), scores end
    return apply_limiter(scores, limiter(selector))
end
