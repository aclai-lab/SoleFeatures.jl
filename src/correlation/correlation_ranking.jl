struct CorrelationRanking <: AbstractCorrelationFilter
    nbest::Int64
    cor_algorithm::Symbol
    memorysaving::Bool

    function CorrelationRanking(k::Int64, cor_algorithm::Symbol, memorysaving::Bool)
        k < 0 && throw(ErrorException("k must be greater or equal 0"))
        if !(cor_algorithm in [:pearson, :spearman, :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(k, cor_algorithm, memorysaving)
    end

    CorrelationRanking(k::Int64, cor_algorithm::Symbol) = new(k, cor_algorithm, false)
end

# traits
is_multivariate(::CorrelationRanking) = true
is_unsupervised(::CorrelationRanking) = true

# getter
nbest(selector::CorrelationRanking) = selector.nbest

function apply(df::AbstractDataFrame, selector::CorrelationRanking)::Vector{Integer}
    k = nbest(selector)
    cormtrx = _buildcormtrx(df, selector)
    bestidxes = findcorrelation(cormtrx)
    return bestidxes[1:k]
end
