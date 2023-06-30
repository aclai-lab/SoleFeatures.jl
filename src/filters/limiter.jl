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
        !(ordf in [>, <, >=, <=, ==]) && throw(DomainError("`ordf`"))
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

"""
Meta limiter: a limiter that applies its property limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least half or more of its elements.
"""
struct MajorityLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
end

"""
# Example

```jldoctest
julia> ml = MajorityLimiter(ThresholdLimiter(1, ==))
MajorityLimiter{ThresholdLimiter}(ThresholdLimiter(1, ==))

julia> v = [ [1,0,0,0], [1,1,0,0], [1,1,1,1] ]
3-element Vector{Vector{Int64}}:
 [1, 0, 0, 0]
 [1, 1, 0, 0]
 [1, 1, 1, 1]

julia> limit(v, ml)
2-element Vector{Int64}:
 2
 3
```
"""
function limit(scores::AbstractVector, ml::MajorityLimiter)
    accepted = length.([ limit(score, ml.limiter) for score in scores ])
    bounds = ceil.(length.(scores) * 0.5)
    return findall(accepted .>= bounds)
end

# ========================================================================================
# AtLeast limiter

"""
Meta limiter: a limiter that applies its property limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least `atleast` elements.
"""
struct AtLeastLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
    atleast::Int
end

"""
# Example

```jldoctest
julia> al = AtLeastLimiter(ThresholdLimiter(0.5, <=), 1)
AtLeastLimiter{ThresholdLimiter}(ThresholdLimiter(0.5, <=), 1)

julia> v = [ [0.2,0,0,0], [5,8,9,7], [1,1,1,1] ]
3-element Vector{Vector{Float64}}:
 [0.2, 0.0, 0.0, 0.0]
 [5.0, 8.0, 9.0, 7.0]
 [1.0, 1.0, 1.0, 1.0]

julia> limit(v, al)
1-element Vector{Int64}:
 1
```
"""
function limit(scores::AbstractVector, al::AtLeastLimiter)
    accepted = length.([ limit(score, al.limiter) for score in scores ])
    return findall(accepted .>= al.atleast)
end
