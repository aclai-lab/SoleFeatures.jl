"""
Here you can create new limiters.

1. Create a concrete type (struct) of an `AbstractLimiter` with needed params
2. Create struct accessors
3. Implement `apply_limiter` function for the new struct
"""

# ========================================================================================
# Threshold limiter

"""
Scores are evaluated by a specified threshold and sorting.
"""
struct ThresholdLimiter <: AbstractFilterLimiter
    threshold::Real
    ordf::Function

    function ThresholdLimiter(threshold::Real, ordf::Function)
        !(ordf in [>, <, >=, <=]) && throw(DomainError("`ordf`"))
        return new(threshold, ordf)
    end
end

threshold(tl::ThresholdLimiter) = tl.threshold
ordf(tl::ThresholdLimiter) = tl.ordf

function apply_limiter(scores::AbstractVector{<:Real}, tl::ThresholdLimiter)
    return findall(ordf(tl)(threshold(tl)), scores)
end

# ========================================================================================
# Ranking limiter

"""
Scores are evaluated by selecting the best first in ascending or descending order
"""
struct RankingLimiter <: AbstractFilterLimiter
    nbest::Int
    rev::Bool

    function RankingLimiter(nbest::Integer, rev::Bool)
        nbest <= 0 && throw(DomainError(nbest, "`nbest` must be > 0"))
        new(nbest, rev)
    end
    RankingLimiter(nbest::Integer) = RankingLimiter(nbest, false)
end

nbest(rl::RankingLimiter) = rl.nbest
rev(rl::RankingLimiter) = rl.rev

function apply_limiter(scores::AbstractVector{<:Real}, rl::RankingLimiter)
    return sortperm(scores; rev=rev(rl))[1:nbest(rl)]
end

# ========================================================================================
# Group limiters (should be an abstract type?)

struct GroupFittestLimiter <: AbstractFilterLimiter
    suiteness::Float64

    function GroupFittestLimiter(suiteness::AbstractFloat)
        !(0.0 < suiteness < 1.0) &&
            throw(DomainError(suiteness, "Must be within 0.0 and 1.0"))
        new(suiteness)
    end
end

struct GroupOneInLimiter <: AbstractFilterLimiter end

suiteness(gfl::GroupFittestLimiter) = gfl.suiteness

function apply_limiter(
    winboard::AbstractVector{BitVector},
    gfl::GroupFittestLimiter
)
    return findall([sum(bm) >= ceil(Int, length(bm) * suiteness(gfl)) for bm in winboard])
end

function apply_limiter(
    winboard::AbstractVector{BitVector},
    goil::GroupOneInLimiter
)
    wins = sum(winboard; dims=1)[1]
    return findall(>=(1), wins)
end
