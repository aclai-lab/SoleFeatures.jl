struct RandomRanking <: AbstractFilterBased
    k::Integer
    rnd_seed::Union{Integer,Nothing}

    RandomRanking(k::Integer) = new(k, nothing)
    RandomRanking(k::Integer, rnd_seed::Integer) = new(k, rnd_seed)
end

selector_k(selector::RandomRanking) = selector.k
selector_rankfunct(selector::RandomRanking) = (ncols, k) ->
                                                StatsBase.sample(1:ncols, k; replace=false)
selector_seed(selector::RandomRanking) = selector.rnd_seed

function build_bitmask(df::AbstractDataFrame, selector::RandomRanking)::BitVector
    n_cols = ncol(df)
    k = selector_k(selector)
    f = selector_rankfunct(selector)
    rnd_seed = selector_seed(selector)
    bm = zeros(n_cols)
    !isnothing(rnd_seed) && Random.seed!(selector_seed(selector)) # set seed if rnd_seed is not nothing
    indices = f(n_cols, k)
    bm[indices] .= true
    return bm
end
