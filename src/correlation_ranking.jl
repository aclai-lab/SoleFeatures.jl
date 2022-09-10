struct CorrelationRanking <: AbstractFilterBased
    k::Int64
    cor_algorithm::Symbol
    memorysaving::Bool

    CorrelationRanking(k::Int64, cor_algorithm::Symbol, memorysaving::Bool) =
        new(k, cor_algorithm, memorysaving)
    function CorrelationRanking(k::Int64, cor_algorithm::Symbol)
        if k < 0
            throw(ErrorException("k must be greater or equal 0"))
        end
        if !(cor_algorithm in [:pearson, :spearman, :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(k, cor_algorithm, false)
    end
end

selector_k(selector::CorrelationRanking) = selector.k
function selector_rankfunct(selector::CorrelationRanking)
    if selector.cor_algorithm == :pearson
        return StatsBase.cor
    elseif selector.cor_algorithm == :spearman
        return StatsBase.corspearman
    elseif selector.cor_algorithm == :kendall
        return StatsBase.corkendall
    end
end
selector_memorysaving(selector::CorrelationRanking) = selector.memorysaving

function build_bitmask(
    df::AbstractDataFrame,
    selector::CorrelationRanking
)::BitVector
    n_cols = ncol(df)
    k = selector_k(selector)
    rf = selector_rankfunct(selector) # rank function
    ms = selector_memorysaving(selector) # memory saving options

    k > n_cols && return trues(ncol) # return immediately if 'k' is greater than columns number

    # compute rank (mean absolute correlation vector)
    ranks = collect(enumerate(correlation(df, rf; memorysaving=ms)))
    # sort ranking
    sort!(ranks, by=x->x[2])
    # prepare bitmask
    bm = falses(n_cols)
    for r in ranks[1:k] # less correlation means good attribute
        bm[r[1]] = true
    end
    return bm
end
