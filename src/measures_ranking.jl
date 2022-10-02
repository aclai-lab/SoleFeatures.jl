struct MeasuresRanking <: AbstractFilterBased
    k::Integer
    measures_selector::AbstractFilterBased
    # measures_selector::Union{AbstractFilterBased,Type{T}} where T

    function MeasuresRanking(k::Integer, ms::AbstractFilterBased)
        typeof(ms) === MeasuresRanking &&
            throw(ErrorException("measures_selector can't be type of " * typeof(ms)))
        new(k, ms)
    end
end

# switch to constants.jl or utils.jl
_MEASURES_NAMES = [ getnames(catch22)..., :mean, :min, :max ]
_MEASURES = Dict{Symbol, Function}(
    :mean => StatsBase.mean,
    :min => minimum,
    :max => maximum,
    (getnames(catch22) .=> catch22)...
)

selector_k(selector::MeasuresRanking) = selector.k
selector_rankfunct(selector::MeasuresRanking) = selector.measures_selector

function build_bitmask(df::AbstractDataFrame, selector::MeasuresRanking)::BitVector
    # TODO: warning if user provide selector with strange parameters
    mrlock = ReentrantLock()
    k = selector_k(selector)
    n_cols = ncol(df)

    k >= n_cols && return trues(n_cols) # return immediately if 'k' is greater than columns number

    # build df for each measure (measures_df is Vecotr of df)
    # measures_df = [ _MEASURES[name].(df) for name in _MEASURES_NAMES ] # On inswectWingbeat 697s
    measures_df = []
    Threads.@threads for name in _MEASURES_NAMES # On inswectWingbeat 392s
        mdf = _MEASURES[name].(df)
        lock(mrlock)
        try
            push!(measures_df, mdf)
        finally
            unlock(mrlock)
        end
    end

    # build bitmasks for each of 25 measure dataframe
    measures_sel = selector_rankfunct(selector)
    ranks = fill(0, n_cols)
    for mdf in measures_df
        ranks .+= build_bitmask(mdf, measures_sel)
    end

    bestidxes = sortperm(ranks; rev=true)[1:k]

    bm = falses(n_cols)
    bm[bestidxes] .= true
    return bm
end
