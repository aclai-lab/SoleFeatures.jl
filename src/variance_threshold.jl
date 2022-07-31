struct VarianceThreshold <: AbstractFilterBased
    threshold::Float64

    function VarianceThreshold(threshold::Float64)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
        new(threshold)
    end
end

selector_threshold(selector::VarianceThreshold) = selector.threshold
selector_function(selector::VarianceThreshold) = StatsBase.var

function build_bitmask(
    df::AbstractDataFrame,
    selector::VarianceThreshold
)::BitVector
    return map(x->(selector_function(selector)(collect(Iterators.flatten(x))) >= selector_threshold(selector)), eachcol(df))
end
