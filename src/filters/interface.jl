############################################################################################
############################################################################################
############################################################################################

abstract type AbstractScoringCriterion end

issupervised(sc::AbstractScoringCriterion) = error("Please, provide method issupervised($(typeof(sc))).")
isunivariate(sc::AbstractScoringCriterion) = error("Please, provide method isunivariate($(typeof(sc))).")
isunsupervised(sc::AbstractScoringCriterion) = !issupervised(sc)
ismultivariate(sc::AbstractScoringCriterion) = !isunivariate(sc)

idxtype(sc::AbstractScoringCriterion) = ismultivariate(sc) ? Integer : Vector{Integer}
scoretype(sc::AbstractScoringCriterion) = error("Please, provide method scoretype($(typeof(sc))).")
infotype(::AbstractScoringCriterion) = nothing

function scores(sc::AbstractScoringCriterion, args...; kwargs...)
    # TODO: provide description for supervised or unsupervised scores methods
    return error("Please, provide method scores($(typeof(sc)), args...; kwargs...).")
end

function apply(sc::AbstractScoringCriterion, X::Dataset; returndetails::Bool)
    !isunsupervised(sc) && error("Provided criteria is not unsupervised")
    return scores(sc, X; returndetails = returndetails)
end

function apply(sc::AbstractScoringCriterion, X::Dataset, y::AbstractVector{<:Class}; returndetails::Bool)
    !issupervised(sc) && error("Provided criteria is not supervised")
    return scores(sc, X, y; returndetails = returndetails)
end

abstract type AbstractScalarCriterion <: AbstractScoringCriterion end

scoretype(::AbstractScalarCriterion) = Number
polarity(sc::AbstractScalarCriterion) = error("Please, provide method polarity($(typeof(sc))).")

abstract type AbstractNonScalarCriterion <: AbstractScoringCriterion end

############################################################################################
############################################################################################
############################################################################################

abstract type ScoreFilterBasedFS <: AbstractFilterBasedFS end

requirescalar(s::ScoreFilterBasedFS) = true

"""
    limit(s, )

return indices of suitable `scores` based on provided `limiter`
"""
function limit(s::ScoreFilterBasedFS, scores; kwargs...)
    return error("Please, provide method limit($(typeof(l)), scores).")
end

function select(s::ScoreFilterBasedFS, sc::AbstractScoringCriterion, X::Dataset, args...; kwarg...)
    requirescalar(s) && !(sc isa AbstractScalarCriterion) &&
        error("$(typeof(s)) needs a scalar criterion")
    return limit(s, apply(sc, X, y, args...; returndetails = false, kwargs...))
end

struct KBestSelector <: ScoreFilterBasedFS
    k::Integer
    function KBestSelector(k::Integer)
        k <= 0 && throw(DomainError("k must be greater than 0"))
        return new(k)
    end
end

k(s::KBestSelector) = s.k

function limit(s::KBestSelector, scores::AbstractVector)

end

struct ThresholdSelector <: ScoreFilterBasedFS
    threshold::Float64
    compf::Function
    function ThresholdSelector(threshold::AbstractFloat, compf::Function)
        !(compf in [>, <, >=, <=, ==]) && throw(DomainError("compf"))
        return new(threshold, compf)
    end
end

threshold(s::ThresholdSelector) = s.threshold
compf(s::ThresholdSelector) = s.compf

function limit(s::ThresholdSelector, scores::AbstractVector)
    return findall(compf(s)(threshold(s)), getindex.(scores, 2))
end
