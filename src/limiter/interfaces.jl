abstract type AbstractLimiter end

abstract type AbstractFilterLimiter <: AbstractLimiter end

abstract type AbstractWrapperLimiter <: AbstractLimiter end

abstract type AbstractEmbeddedLimiter <: AbstractLimiter end

struct ThresholdLimiter <: AbstractFilterLimiter
    threshold::Float64
    ordf::Function

    function ThresholdLimiter(threshold::AbstractFloat, ordf::Function)
        !(ordf in [>, <, >=, <=]) && throw(DomainError("`ford`"))
        return new(threshold, ordf)
    end
end

# ---------- testing

struct RankingLimiter <: AbstractFilterLimiter
    nbest::Int64
    rev::Bool

    function RankingLimiter(nbest::Integer, rev::Bool)
        nbest <= 0 && throw(DomainError(nbest, "`nbest` must be > 0"))
        new(nbest, rev)
    end
    RankingLimiter(nbest::Integer) = RankingLimiter(nbest, false)
end

struct FittestLimiter <: AbstractFilterLimiter
    suiteness::Float64

    function FittestLimiter(suiteness::AbstractFloat)
        !(0.0 < suiteness < 1.0) &&
            throw(DomainError(suiteness, "Must be within 0.0 and 1.0"))
        new(suiteness)
    end
end

struct OneInLimiter <: AbstractFilterLimiter end
