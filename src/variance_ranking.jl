struct VarianceRanking <: AbstractFilterBased
    k::Integer

    function VarianceRanking(k::Integer)
        @assert k >= 0 "k must be greater or equal 0"
        new(k)
    end
end

selector_k(selector::VarianceRanking) = selector.k

function build_bitmask(df::AbstractDataFrame, selector::VarianceRanking)::BitVector
    n_cols = ncol(df)
    k = selector_k(selector)

    k >= n_cols && return trues(n_cols)

    vars = [ StatsBase.var(Iterators.flatten(c)) for c in eachcol(df) ]
    replace!(vars, NaN => -Inf)
    bestidxes = sortperm(vars; rev=true)[1:k]

    bm = falses(n_cols)
    bm[bestidxes] .= true
    return bm
end
