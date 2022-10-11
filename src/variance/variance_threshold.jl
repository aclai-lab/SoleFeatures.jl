struct VarianceThreshold <: AbstractFilterBased
    threshold::AbstractFloat

    function VarianceThreshold(threshold::AbstractFloat)
        threshold < 0.0 &&
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        new(threshold)
    end
end

# traits
is_univariate(::VarianceThreshold) = true
is_unsupervised(::VarianceThreshold) = true

# getter
threshold(selector::VarianceThreshold) = selector.threshold

function apply(df::AbstractDataFrame, selector::VarianceThreshold)::Vector{Integer}
    th = threshold(selector)
    vars = [ StatsBase.var(Iterators.flatten(c)) for c in eachcol(df) ]
    return findall(>=(th), vars)
end
