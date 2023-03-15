struct CorrelationFilter{T <: AbstractFilterLimiter} <: AbstractCorrelationFilter{T}
    limiter::T
    # parameters
    corf::Function
    memorysaving::Bool

    function CorrelationFilter(
        limiter::T,
        corf::Function,
        memorysaving::Bool = false
    ) where {T <: AbstractFilterLimiter}
        return new{T}(limiter, corf, memorysaving)
    end
end

# limiter(selector::CorrelationFilter) = selector.limiter
corf(selector::CorrelationFilter) = selector.corf
memorysaving(selector::CorrelationFilter) = selector.memorysaving

# traits
is_unsupervised(::AbstractCorrelationFilter{<:AbstractFilterLimiter}) = true
is_multivariate(::AbstractCorrelationFilter{<:AbstractFilterLimiter}) = true

# Ranking constructor

function CorrelationRanking(
    nbest::Integer,
    corf::Function,
    memorysaving::Bool = false
)
    return CorrelationFilter(RankingLimiter(nbest), corf, memorysaving)
end

function apply(
    X::AbstractDataFrame,
    selector::CorrelationFilter{RankingLimiter}
)::Vector{Integer}
    k = nbest(limiter(selector))
    cormtrx = _buildcormtrx(X, selector)
    bestidxes = findcorrelation(cormtrx)
    return bestidxes[1:k]
end

# Threshold constructor

function CorrelationThreshold(
    threshold::AbstractFloat,
    corf::Function,
    memorysaving::Bool = true
)
    return CorrelationFilter(ThresholdLimiter(threshold, <=), corf, memorysaving)
end

function apply(
    X::AbstractDataFrame,
    selector::CorrelationFilter{ThresholdLimiter}
)::Vector{Integer}
    thr = threshold(limiter(selector))
    cormtrx = _buildcormtrx(X, selector)
    bestidxes = findcorrelation(cormtrx; threshold=thr)
    return bestidxes
end
