abstract type AbstractLimiter end

"""
    limit(scores, l)

return indices of best scores
"""
function limit(scores, l::AbstractLimiter)
    return error("`limit` not implemented for type: $(typeof(l))")
end

# ========================================================================================
# Identity limiter

struct IdentityLimiter <: AbstractLimiter end

function limit(scores, il::IdentityLimiter)
    return collect(1:length(scores))
end

# ========================================================================================
# Threshold limiter

"""
Scores are evaluated by a specified threshold and sorting.
"""
struct ThresholdLimiter <: AbstractLimiter
    threshold::Real
    ordf::Function

    function ThresholdLimiter(threshold::Real, ordf::Function)
        !(ordf in [>, <, >=, <=]) && throw(DomainError("`ordf`"))
        return new(threshold, ordf)
    end
end

threshold(tl::ThresholdLimiter) = tl.threshold
ordf(tl::ThresholdLimiter) = tl.ordf

function limit(scores::AbstractVector{<:Real}, tl::ThresholdLimiter)
    return findall(ordf(tl)(threshold(tl)), scores)
end

# ========================================================================================
# Ranking limiter

"""
Scores are evaluated by selecting the best first in ascending or descending order
"""
struct RankingLimiter <: AbstractLimiter
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

function limit(scores::AbstractVector{<:Real}, rl::RankingLimiter)
    return sortperm(scores; rev=rev(rl))[1:nbest(rl)]
end

# ========================================================================================
# Majority limiter

struct MajorityLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
end

function limit(scores::AbstractVector{<:AbstractVector}, ml::MajorityLimiter)
    accepted = length.([ limit(score, ml.limiter) for score in scores ])
    bounds = ceil.(length.(scores) * 0.5)
    return findall(accepted .>= bounds)
end

# ========================================================================================
# AtLeast limiter

struct AtLeastLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
    atleast::Int
end

function limit(scores::AbstractVector{<:AbstractVector}, al::AtLeastLimiter)
    res = length.([ limit(score, al.limiter) for score in scores ])
    return findall(res .>= al.atleast)
end
