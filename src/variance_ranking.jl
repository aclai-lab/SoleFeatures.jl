struct VarianceRanking <: AbstractFilterBased
    k::Int64

    function VarianceRanking(k::Int64)
        @assert k >= 0 "k must be greater or equal 0"
        new(k)
    end
end

selector_k(selector::VarianceRanking) = selector.k
selector_rankfunct(selector::VarianceRanking) = StatsBase.var

function build_bitmask(df::AbstractDataFrame, selector::VarianceRanking)::BitVector
    ranks = map(x->(x[1], selector_rankfunct(selector)(collect(Iterators.flatten(x[2])))), enumerate(eachcol(df)))

    # TODO: improve NaN management
    function lt(x, y)
        if isnan(x) && !isnan(y)
            return true
        elseif !isnan(x) && isnan(y)
            return false
        end
        return x < y
    end
    sort!(ranks, by=x->x[2], lt=lt)

    n_cols = ncol(df)
    if selector_k(selector) < n_cols
        bm = falses(n_cols)
        lower = n_cols - selector_k(selector) + 1
        for r in ranks[lower:n_cols]
            bm[r[1]] = true
        end
    else
        bm = trues(n_cols)
    end
    return bm
end
