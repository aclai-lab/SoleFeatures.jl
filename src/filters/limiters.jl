# ---------------------------------------------------------------------------- #
#                            functions definitions                             #
# ---------------------------------------------------------------------------- #
"""
    limit(scores, l)

return indices of suitable `scores` based on provided `limiter`
"""
# function limit(scores::Any, l::AbstractLimiter)
#     return error("`limit` not implemented for type: $(typeof(l))")
# end

# (l::AbstractLimiter)(scores::Any) = limit(scores, l)

# struct FilterLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
#     filter :: AbstractFilter{F,T,L,D}
#     rank   :: Int64
# end

# ---------------------------------------------------------------------------- #
#                             threshold limiter                                #
# ---------------------------------------------------------------------------- #
"""
Scores are evaluated by a specified threshold and sorting.
"""
struct ThresholdLimiterInfo <: AbstractLimiterInfo
    threshold :: Real
    ordf      :: Function

    function ThresholdLimiterInfo(threshold::Real, ordf::Function)
        valid_ops = (>, <, ≥, ≤, ≡)
        ordf ∈ valid_ops || throw(DomainError("`ordf`"))
        new(threshold, ordf)
    end
end

function Base.show(io::IO, info::ThresholdLimiterInfo)
    println(io, "$(typeof(info))")
    println(io, "  threshold = '$(info.ordf) $(info.threshold)'")
end

struct ThresholdLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
    filter :: AbstractFilter{F,T,L,D}
    rank   :: Vector{Int64}
    info   :: ThresholdLimiterInfo

    function ThresholdLimiter(
        filter    :: AbstractFilter{F,T,L,D};
        ordf      :: Function=(≥),
        threshold :: Real
    ) where {F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality}
        scores = get_score(filter)
        revflag = ordf ∈ (>, ≥)
        sorted_indices = sortperm(scores; rev=revflag)
        rank = sorted_indices[findall(ordf.(scores[sorted_indices], threshold))]

        info = ThresholdLimiterInfo(threshold, ordf)

        return new{F,T,L,D}(filter, rank, info)
    end
end

# threshold(tl::ThresholdLimiter) = tl.threshold
# ordf(tl::ThresholdLimiter) = tl.ordf

# function limit(scores::AbstractVector{<:Real}, tl::ThresholdLimiter)
#     return findall(ordf(tl)(threshold(tl)), scores)
# end

# function limit(scores::AbstractVector{GroupScore}, tl::ThresholdLimiter)
#     return limit([s.score for s in scores], tl)
# end

# # ---------------------------------------------------------------------------- #
# #                              ranking limiter                                 #
# # ---------------------------------------------------------------------------- #
# """
# Scores are evaluated by selecting the best first in ascending or descending order
# """
# struct RankingLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
#     nbest::Int
#     rev::Bool

#     function RankingLimiter(nbest::Integer, rev::Bool)
#         nbest > 0 || throw(DomainError(nbest, "`nbest` must be > 0"))
#         new(nbest, rev)
#     end
#     RankingLimiter(nbest::Integer) = RankingLimiter(nbest, false)
# end

# nbest(rl::RankingLimiter) = rl.nbest
# rev(rl::RankingLimiter) = rl.rev

# function limit(scores::AbstractVector{<:Real}, rl::RankingLimiter)
#     return sortperm(scores; rev=rev(rl))[1:nbest(rl)]
# end

# function limit(scores::AbstractVector{GroupScore}, rl::RankingLimiter)
#     return limit([s.score for s in scores], rl)
# end

# # ---------------------------------------------------------------------------- #
# #                              majority limiter                                #
# # ---------------------------------------------------------------------------- #
# """
# Meta limiter: a limiter that applies its property limiter for each element in scores.
# An item in scores is accepted only if the property limiter selects at least half or more of its elements.

# # Example
# ```jldoctest
# julia> ml = MajorityLimiter(ThresholdLimiter(1, ==))
# MajorityLimiter{ThresholdLimiter}(ThresholdLimiter(1, ==))

# julia> v = [ [1,0,0,0], [1,1,0,0], [1,1,1,1] ]
# 3-element Vector{Vector{Int64}}:
#  [1, 0, 0, 0]
#  [1, 1, 0, 0]
#  [1, 1, 1, 1]

# julia> limit(v, ml)
# 2-element Vector{Int64}:
#  2
#  3
# ```
# """
# struct MajorityLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
#     limiter::T
# end

# function limit(scores::AbstractVector, ml::MajorityLimiter)
#     accepted = length.([limit(score, ml.limiter) for score in scores])
#     # change length(scores) with getindex.(size.(scores, 1), 1)
#     bounds = ceil.(length.(scores) * 0.5)
#     return findall(accepted .≥ bounds)
# end

# function limit(scores::AbstractVector{GroupScore}, ml::MajorityLimiter)
#     return limit([s.score for s in scores], ml)
# end

# # ---------------------------------------------------------------------------- #
# #                               atleast limiter                                #
# # ---------------------------------------------------------------------------- #
# """
# Meta limiter: a limiter that applies its property limiter for each element in scores.
# An item in scores is accepted only if the property limiter selects at least `atleast` elements.

# # Example
# ```jldoctest
# julia> al = AtLeastLimiter(ThresholdLimiter(0.5, <=), 1)
# AtLeastLimiter{ThresholdLimiter}(ThresholdLimiter(0.5, <=), 1)

# julia> v = [ [0.2,0,0,0], [5,8,9,7], [1,1,1,1] ]
# 3-element Vector{Vector{Float64}}:
#  [0.2, 0.0, 0.0, 0.0]
#  [5.0, 8.0, 9.0, 7.0]
#  [1.0, 1.0, 1.0, 1.0]

# julia> limit(v, al)
# 1-element Vector{Int64}:
#  1
# ```
# """
# struct AtLeastLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
#     limiter::T
#     atleast::Int
# end

# function limit(scores::AbstractVector, al::AtLeastLimiter)
#     accepted = length.([limit(score, al.limiter) for score in scores])
#     return findall(accepted .≥ al.atleast)
# end

# function limit(scores::AbstractVector{GroupScore}, al::AtLeastLimiter)
#     return limit([s.score for s in scores], al)
# end

# # ---------------------------------------------------------------------------- #
# #                             percentange limiter                              #
# # ---------------------------------------------------------------------------- #
# """
# `PercentageLimiter` is an implementation of an `AbstractLimiter` which
# limits the selection to a fraction of the available variables.
# """
# struct PercentageLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
#     perc::Float64
#     rev::Bool

#     function PercentageLimiter(perc::AbstractFloat, rev::Bool)
#         (0 ≤ perc ≤ 1.0) || throw(DomainError(perc, "`perc` must be ≥ 0 and ≤ 1"))
#         new(perc, rev)
#     end
#     PercentageLimiter(perc::AbstractFloat) = PercentageLimiter(perc, true)
# end

# """
#     perc(pl)

# Retrieve the fraction (percentage / 100) of variables that will be
# selected with `pl` limiter.
# """
# perc(pl::PercentageLimiter) = pl.perc

# """
#     rev(pl)

# Return whether the selection is reverse or not. It follows the same
# semantic of `rev` parameter of the function [`sort`](@ref).
# """
# rev(pl::PercentageLimiter) = pl.rev

# function limit(scores::AbstractVector{<:Real}, l::PercentageLimiter)
#     len = Int(ceil(length(scores) * perc(l)))
#     return sortperm(scores; rev = rev(l))[1:len]
# end

# function limit(scores::AbstractVector{GroupScore}, l::PercentageLimiter)
#     return limit([s.score for s in scores], l)
# end