# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
abstract type AbstractLimiter end

# ---------------------------------------------------------------------------- #
#                            functions definitions                             #
# ---------------------------------------------------------------------------- #
"""
    limit(scores, l)

return indices of suitable `scores` based on provided `limiter`
"""
function limit(scores::Any, l::AbstractLimiter)
    return error("`limit` not implemented for type: $(typeof(l))")
end

(l::AbstractLimiter)(scores::Any) = limit(scores, l)

# ---------------------------------------------------------------------------- #
#                              identity limiter                                #
# ---------------------------------------------------------------------------- #
struct IdentityLimiter <: AbstractLimiter end

function limit(scores, il::IdentityLimiter)
    return collect(1:length(scores))
end

# ---------------------------------------------------------------------------- #
#                             threshold limiter                                #
# ---------------------------------------------------------------------------- #
"""
Scores are evaluated by a specified threshold and sorting.
"""
struct ThresholdLimiter <: AbstractLimiter
    threshold::Real
    ordf::Function

    function ThresholdLimiter(threshold::Real, ordf::Function)
        valid_ops = (>, <, ≥, ≤, ≡)
        ordf ∈ valid_ops || throw(DomainError("`ordf`"))
        return new(threshold, ordf)
    end
end

threshold(tl::ThresholdLimiter) = tl.threshold
ordf(tl::ThresholdLimiter) = tl.ordf

function limit(scores::AbstractVector{<:Real}, tl::ThresholdLimiter)
    return findall(ordf(tl)(threshold(tl)), scores)
end

# ---------------------------------------------------------------------------- #
#                              ranking limiter                                 #
# ---------------------------------------------------------------------------- #
"""
Scores are evaluated by selecting the best first in ascending or descending order
"""
struct RankingLimiter <: AbstractLimiter
    nbest::Int
    rev::Bool

    function RankingLimiter(nbest::Integer, rev::Bool)
        nbest > 0 || throw(DomainError(nbest, "`nbest` must be > 0"))
        new(nbest, rev)
    end
    RankingLimiter(nbest::Integer) = RankingLimiter(nbest, false)
end

nbest(rl::RankingLimiter) = rl.nbest
rev(rl::RankingLimiter) = rl.rev

function limit(scores::AbstractVector{<:Real}, rl::RankingLimiter)
    return sortperm(scores; rev=rev(rl))[1:nbest(rl)]
end

# ---------------------------------------------------------------------------- #
#                              majority limiter                                #
# ---------------------------------------------------------------------------- #
"""
Meta limiter: a limiter that applies its property limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least half or more of its elements.

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
struct MajorityLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
end

function limit(scores::AbstractVector, ml::MajorityLimiter)
    accepted = length.([limit(score, ml.limiter) for score in scores])
    # change length(scores) with getindex.(size.(scores, 1), 1)
    bounds = ceil.(length.(scores) * 0.5)
    return findall(accepted .≥ bounds)
end

# ---------------------------------------------------------------------------- #
#                               atleast limiter                                #
# ---------------------------------------------------------------------------- #
"""
Meta limiter: a limiter that applies its property limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least `atleast` elements.

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
struct AtLeastLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
    atleast::Int
end

function limit(scores::AbstractVector, al::AtLeastLimiter)
    accepted = length.([limit(score, al.limiter) for score in scores])
    return findall(accepted .≥ al.atleast)
end

# ---------------------------------------------------------------------------- #
#                             percentange limiter                              #
# ---------------------------------------------------------------------------- #
"""
`PercentageLimiter` is an implementation of an `AbstractLimiter` which
limits the selection to a fraction of the available variables.
"""
struct PercentageLimiter <: AbstractLimiter
    perc::Float64
    rev::Bool

    function PercentageLimiter(perc::AbstractFloat, rev::Bool)
        (0 ≤ perc ≤ 1.0) || throw(DomainError(perc, "`perc` must be ≥ 0 and ≤ 1"))
        new(perc, rev)
    end
    PercentageLimiter(perc::AbstractFloat) = PercentageLimiter(perc, true)
end

"""
    perc(pl)

Retrieve the fraction (percentage / 100) of variables that will be
selected with `pl` limiter.
"""
perc(pl::PercentageLimiter) = pl.perc

"""
    rev(pl)

Return whether the selection is reverse or not. It follows the same
semantic of `rev` parameter of the function [`sort`](@ref).
"""
rev(pl::PercentageLimiter) = pl.rev

function limit(scores::AbstractVector{<:Real}, l::PercentageLimiter)
    len = Int(ceil(length(scores) * perc(l)))
    return sortperm(scores; rev = rev(l))[1:len]
end
