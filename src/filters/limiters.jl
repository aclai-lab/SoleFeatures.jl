# ---------------------------------------------------------------------------- #
#                             threshold limiter                                #
# ---------------------------------------------------------------------------- #
struct ThresholdLimiterInfo <: AbstractLimiterInfo
    threshold :: Real
    ordf      :: Function

    function ThresholdLimiterInfo(threshold::Real, ordf::Function)
        valid_ops = (>, <, ≥, ≤)
        ordf ∈ valid_ops || throw(DomainError("`ordf`"))
        new(threshold, ordf)
    end
end

function Base.show(io::IO, info::ThresholdLimiterInfo)
    println(io, "$(typeof(info))")
    println(io, "  threshold = '$(info.ordf) $(info.threshold)'")
end

"""
    ThresholdLimiter(filter::AbstractFilter; ordf::Function, threshold::Real)

A limiter that selects features based on a threshold comparison of their scores.

This limiter evaluates the scores from a filter and returns the indices of features
whose scores satisfy the threshold condition defined by the comparison operator `ordf`.

# Arguments
- `filter::AbstractFilter`: The filter whose scores will be evaluated
- `ordf::Function`: Comparison operator for threshold evaluation. Valid operators are:
  - `(>, <, ≥, ≤)`
- `threshold::Real`: The threshold value to compare against

# Sorting Behavior
- For `>` and `≥`: Sorts in **descending order** (`rev=true`), placing highest scores first
- For `<` and `≤`: Sorts in **ascending order** (`rev=false`), placing lowest scores first

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats = Chi2Filter(X, y)

# apply threshold filter
ThresholdLimiter(chi2stats; ordf=(>), threshold=7)
```
"""
struct ThresholdLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
    filter :: AbstractFilter{F,T,L,D}
    rank   :: Vector{Int64}
    info   :: ThresholdLimiterInfo

    function ThresholdLimiter(
        filter    :: AbstractFilter{F,T,L,D};
        ordf      :: Function=(≥),
        threshold :: Real
    ) where {F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality}
        info = ThresholdLimiterInfo(threshold, ordf)

        scores = get_score(filter)
        revflag = ordf ∈ (>, ≥)
        sorted_indices = sortperm(scores; rev=revflag)
        rank = sorted_indices[findall(ordf.(scores[sorted_indices], threshold))]

        new{F,T,L,D}(filter, rank, info)
    end
end

# ---------------------------------------------------------------------------- #
#                              ranking limiter                                 #
# ---------------------------------------------------------------------------- #
struct RankingLimiterInfo <: AbstractLimiterInfo
    nbest :: Int64
    rev   :: Bool

    function RankingLimiterInfo(nbest::Int64, rev::Bool)
        nbest > 0 || throw(DomainError(nbest, "`nbest` must be > 0"))

        new(nbest, rev)
    end
end

"""
    RankingLimiter(filter::AbstractFilter; nbest::Int64, rev::Bool=true)

A limiter that selects the top `nbest` features according to their scores, 
using either descending or ascending order.

This limiter evaluates the scores from a filter and returns the indices of the 
`nbest` features with the highest (or lowest, if `rev=false`) scores.

# Arguments
- `filter::AbstractFilter`: The filter whose scores will be evaluated
- `nbest::Int64`: Number of top features to select (must be > 0)
- `rev::Bool`:
  If `true` (default), selects the highest scores (descending order). 
  If `false`, selects the lowest scores (ascending order).

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats = Chi2Filter(X, y)

# select top 2 features with highest scores
RankingLimiter(chi2stats; nbest=2)

# select top 2 features with lowest scores
RankingLimiter(chi2stats; nbest=2, rev=false)
```
"""
struct RankingLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
    filter :: AbstractFilter{F,T,L,D}
    rank   :: Vector{Int64}
    info   :: RankingLimiterInfo

    function RankingLimiter(
        filter :: AbstractFilter{F,T,L,D};
        nbest  :: Int64,
        rev    :: Bool=true
    ) where {F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality}
        info = RankingLimiterInfo(nbest, rev)

        scores = get_score(filter)
        rank = sortperm(scores; rev)[1:nbest]

        new{F,T,L,D}(filter, rank, info)
    end
end

# ---------------------------------------------------------------------------- #
#                             percentange limiter                              #
# ---------------------------------------------------------------------------- #
struct PercentageLimiterInfo <: AbstractLimiterInfo
    perc :: Real
    rev  :: Bool

    function PercentageLimiterInfo(perc::Real, rev::Bool)
        (0 ≤ perc ≤ 1.0) || throw(DomainError(perc, "`perc` must be ≥ 0 and ≤ 1"))
        new(perc, rev)
    end
end

"""
    PercentageLimiter(filter::AbstractFilter; perc::Real, rev::Bool=true)

Selects the top fraction of features according to their scores lenght.

# Arguments
- `filter::AbstractFilter`: The filter whose scores will be evaluated
- `perc::Real`: Fraction of features to keep (0 ≤ perc ≤ 1)
- `rev::Bool`: If `true` (default), sorts descending; if `false`, ascending

# Examples
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats = Chi2Filter(X, y)

# keep top 50% highest-scoring features
PercentageLimiter(chi2stats; perc=0.5)

# keep top 30% lowest-scoring features
PercentageLimiter(chi2stats; perc=0.3, rev=false)
```
"""
struct PercentageLimiter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractLimiter{F,T,L,D}
    filter :: AbstractFilter{F,T,L,D}
    rank   :: Vector{Int64}
    info   :: PercentageLimiterInfo

    function PercentageLimiter(
        filter :: AbstractFilter{F,T,L,D};
        perc   :: Float64,
        rev    :: Bool=true
    ) where {F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality}
        info = PercentageLimiterInfo(perc, rev)

        scores = get_score(filter)
        sorted_indices = sortperm(scores; rev=rev)
        k = Int64(ceil(length(scores) * perc))
        rank = k == 0 ? Int64[] : sorted_indices[1:k]

        new{F,T,L,D}(filter, rank, info)
    end
end

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