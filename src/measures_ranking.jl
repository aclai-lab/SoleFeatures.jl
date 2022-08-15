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
_MEASURES_NAMES = [getnames(catch22)..., :mean, :min, :max]
_MEASURES = Dict()
[ push!(_MEASURES, name => catch22[name]) for name in getnames(catch22) ]
push!(_MEASURES, :mean => StatsBase.mean)
push!(_MEASURES, :min => minimum)
push!(_MEASURES, :max => maximum)

selector_k(selector::MeasuresRanking) = selector.k
selector_rankfunct(selector::MeasuresRanking) = selector.measures_selector

function build_bitmask(df::AbstractDataFrame, selector::MeasuresRanking)::BitVector
    # TODO: warning if user provide selector with strange parameters
    k = selector_k(selector)
    n_cols = ncol(df)

    k > n_cols && return trues(ncol) # return immediately if 'k' is greater than columns number

    # build df for each measure (measures_df is Vecotr of df)
    measures_df = [ _MEASURES[name].(df) for name in _MEASURES_NAMES ]
    # build bitmasks for each of 25 measure dataframe
    measures_sel = selector_rankfunct(selector)
    measures_bm = [ build_bitmask(mdf, measures_sel) for mdf in measures_df ]
    # compute ranking for each attribute
    ranks = sum(measures_bm, dims=1)[1]
    ranks = collect(enumerate(ranks))
    # sort rankings
    sort!(ranks; by=x->x[2], rev=true)
    # prepare bitmask
    bm = falses(n_cols)
    for r in ranks[1:k]
        bm[r[1]] = true
    end
    return bm
end
