###########################################################################################
###########################################################################################
###########################################################################################

abstract type AbstractScoringCriterion end

issupervised(sc::AbstractScoringCriterion) =
    error("Please, provide method issupervised($(typeof(sc))).")
isunivariate(sc::AbstractScoringCriterion) =
    error("Please, provide method isunivariate($(typeof(sc))).")
isunsupervised(sc::AbstractScoringCriterion) = !issupervised(sc)
ismultivariate(sc::AbstractScoringCriterion) = !isunivariate(sc)

idxtype(sc::AbstractScoringCriterion) = ismultivariate(sc) ? Integer : Vector{Integer}
scoretype(sc::AbstractScoringCriterion) =
    error("Please, provide method scoretype($(typeof(sc))).")
detailstype(::AbstractScoringCriterion) = Nothing

function scores(sc::AbstractScoringCriterion, args...; kwargs...)
    # TODO: provide description for supervised or unsupervised scores methods
    return error("Please, provide method scores($(typeof(sc)), args...; kwargs...).")
end

function apply(sc::AbstractScoringCriterion, X::Dataset; kwargs...)
    !isunsupervised(sc) && error("Provided criteria is not unsupervised")
    return scores(sc, X; kwargs...)
end

function apply(
    sc::AbstractScoringCriterion,
    X::Dataset,
    y::AbstractVector{<:Class};
    kwargs...
)
    !issupervised(sc) && error("Provided criteria is not supervised")
    return scores(sc, X, y; kwargs...)
end

abstract type AbstractScalarCriterion <: AbstractScoringCriterion end

scoretype(::AbstractScalarCriterion) = Number
# polarity(sc::AbstractScalarCriterion) = error("Please, provide method polarity($(typeof(sc))).")

abstract type AbstractNonScalarCriterion <: AbstractScoringCriterion end

###########################################################################################
###########################################################################################
###########################################################################################

abstract type AbstractLimiter end

"""
    limit(s, )

return indices of suitable `scores` based on provided `limiter`
"""
function limit(l::AbstractLimiter, scores; kwargs...)
    return error("Please, provide method limit($(typeof(l)), $(typeof(scores))).")
end

(l::AbstractLimiter)(args...; kwargs...) = limit(l, args...; kwargs...)

requirescalar(l::AbstractLimiter) = true

struct KBestLimiter <: AbstractLimiter
    k::Int
    function KBestLimiter(k::Integer)
        k <= 0 && throw(DomainError("k must be greater than 0"))
        return new(k)
    end
end

k(l::KBestLimiter) = l.k

function limit(l::KBestLimiter, scores::AbstractVector)
    return sortperm(scores; rev = true)[1:k(l)]
end

struct ThresholdLimiter <: AbstractLimiter
    threshold::Float64
    compf::Function
    function ThresholdLimiter(threshold::AbstractFloat, compf::Function)
        !(compf in [>, <, >=, <=, ==]) && throw(DomainError("compf"))
        return new(threshold, compf)
    end
end

threshold(s::ThresholdLimiter) = s.threshold
compf(s::ThresholdLimiter) = s.compf

function limit(l::ThresholdLimiter, scores::AbstractVector)
    return findall(compf(s)(threshold(l)), getindex.(scores, 2))
end

"""
Meta limiter: a limiter that applies its `evaluator` limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least half or more of its elements.
"""
struct MajorityLimiter{T<:AbstractLimiter} <: AbstractLimiter
    evaluator::T
end

evaluator(l::MajorityLimiter) = l.evaluator

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
function limit(l::MajorityLimiter, scores::AbstractVector)
    # accepted = length.(map(evaluator(l), scores))
    accepted = length.([ evaluator(l)(score) for score in scores ])
    # change length(scores) with getindex.(size.(scores, 1), 1)
    bounds = ceil.(length.(scores) * 0.5)
    return findall(accepted .>= bounds)
end

# ========================================================================================
# AtLeast limiter

"""
Meta limiter: a limiter that applies its `evaluator` limiter for each element in scores.
An item in scores is accepted only if the property limiter selects at least `atleast` elements.
"""
struct AtLeastLimiter{T<:AbstractLimiter} <: AbstractLimiter
    evaluator::T
    atleast::Int
end

evaluator(l::AtLeastLimiter) = l.evaluator
atleat(l::AtLeastLimiter) = l.atleast

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
function limit(l::AtLeastLimiter, scores::AbstractVector)
    # accepted = length.(map(evaluator(l), scores))
    accepted = length.([ evaluator(l)(score) for score in scores ])
    return findall(accepted .>= atleat(l))
end

# struct AcceptanceLimiter <: AbstractLimiter
#     l::AbstractLimiter
#     bound::Union{Function, Int}
# end

# evaluator(l::AcceptanceLimiter) = l.evaluator
# bound(l::AcceptanceLimiter) = l.bound

# requirescalar(l::AcceptanceLimiter) = false

# function limit(l::AcceptanceLimiter, scores::AbstractVector)
# end

###########################################################################################
###########################################################################################
###########################################################################################

struct FilterSelector <: AbstractFilterBasedFS
    criterion::AbstractScoringCriterion
    limiter::AbstractLimiter

    function FilterSelector(sc::AbstractScoringCriterion, l::AbstractLimiter)
        if (requirescalar(l) && !(typeof(sc) <: AbstractScalarCriterion))
            throw(ErrorException("Limiter of type $(typeof(l)) require a
                scalar criterion"))
        end
        new(sc, l)
    end
end

criterion(s::FilterSelector) = s.criterion
limiter(s::FilterSelector) = s.limiter

# TOFIX: Bug when using "returndetails = true"
function select(s::FilterSelector, X::Dataset, args...; kwargs...)
    rd = get(kwargs, :returndetais, false)
    res = apply(criterion(s), X, args...; kwargs...)
    if (rd)
        goodidxes = limiter(s)(res[1])
        return (goodidxes, getindex.(res, [goodidxes])...)
    else
        goodidxes = limiter(s)(res)
        return goodidxes
    end
end

(s::FilterSelector)(args...; kwargs...) = select(s, args...; kwargs...)
