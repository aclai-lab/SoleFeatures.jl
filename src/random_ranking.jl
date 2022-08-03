struct RandomRanking <: AbstractFilterBased
    k::Integer
end

selector_k(selector::RandomRanking) = selector.k
selector_rankfunct(selector::RandomRanking) = (ncols, k) -> StatsBase.sample(1:ncols, k; replace=false)

function build_bitmask(df::AbstractDataFrame, selector::RandomRanking)::BitVector
    n_cols = ncol(df)
    k = selector_k(selector)
    f = selector_rankfunct(selector)
    bm = zeros(n_cols)
    indices = f(n_cols, k)
    bm[indices] .= true
    return bm
end
