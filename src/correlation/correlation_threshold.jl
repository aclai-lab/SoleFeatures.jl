struct CorrelationThreshold <: AbstractCorrelationFilter
    threshold::Float64
    cor_algorithm::Symbol
    memorysaving::Bool

    function CorrelationThreshold(threshold::Float64, cor_algorithm::Symbol, memorysaving::Bool)
        threshold < 0.0 &&
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        !(cor_algorithm in [:pearson :spearman :kendall]) &&
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        new(threshold, cor_algorithm, memorysaving)
    end

    function CorrelationThreshold(threshold::Float64, cor_algorithm::Symbol)
        new(threshold, cor_algorithm, false)
    end
end

threshold(selector::CorrelationThreshold) = selector.threshold

function apply(df::AbstractDataFrame, selector::CorrelationThreshold)::Vector{Integer}
    thr = threshold(selector)
    cormtrx = _buildcormtrx(df, selector)
    bestidxes = findcorrelation(cormtrx; threshold=thr)
    return bestidxes
end
