struct VarianceThreshold <: AbstractFilterBased
    threshold::AbstractFloat

    function VarianceThreshold(threshold::AbstractFloat)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
        new(threshold)
    end
end

threshold(selector::VarianceThreshold) = selector.threshold

function build_bitmask(df::AbstractDataFrame, selector::VarianceThreshold)::BitVector
    th = threshold(selector)
    bm = [ StatsBase.var(Iterators.flatten(c)) >= th for c in eachcol(df) ]
    return bm
end
