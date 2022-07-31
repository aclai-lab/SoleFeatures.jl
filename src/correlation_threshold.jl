struct CorrelationThreshold <: AbstractFilterBased
    threshold::Float64
    cor_algorithm::Symbol

    function CorrelationThreshold(threshold::Float64, cor_algorithm::Symbol)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
        if !(cor_algorithm in [:pearson :spearman :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(threshold, cor_algorithm)
    end
end

selector_threshold(selector::CorrelationThreshold) = selector.threshold
function selector_function(selector::CorrelationThreshold)
    if selector.cor_algorithm == :pearson
        return StatsBase.cor
    elseif selector.cor_algorithm == :spearman
        return StatsBase.corspearman
    elseif selector.cor_algorithm == :kendall
        return StatsBase.corkendall
    end
end

function build_bitmask(
    df::AbstractDataFrame,
    selector::CorrelationThreshold
)::BitVector
    avg_vector = correlation(df, selector_function(selector))
    return BitVector(map(x->(x <= selector_threshold(selector)), avg_vector))
end
