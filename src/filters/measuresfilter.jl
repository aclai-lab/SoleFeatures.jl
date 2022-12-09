struct MeasuresFilter{
    T <: AbstractFilterLimiter,
    R <: AbstractLimiter
} <: AbstractMeasuresFilter{T}
    limiter::T
    # parameters
    usedselector::AbstractFeaturesSelector{R}
end

# switch to constants.jl or utils.jl
_MEASURES_NAMES = [ getnames(catch22)..., :mean, :min, :max ]
_MEASURES = Dict{Symbol, Function}(
    :mean => StatsBase.mean,
    :min => minimum,
    :max => maximum,
    (getnames(catch22) .=> catch22)...
)

limiter(selector::MeasuresFilter) = selector.limiter
usedselector(selector::MeasuresFilter) = selector.usedselector

# traits

is_multivariate(::AbstractMeasuresFilter{AbstractFilterLimiter}) = true
is_unsupervised(::AbstractMeasuresFilter{AbstractFilterLimiter}) = true

# Measures Ranking

function MeasuresRanking(
    nbest::Integer,
    selector::AbstractFeaturesSelector{T}
) where {T <: AbstractLimiter}
    return MeasuresFilter(RankingLimiter(nbest, true), selector)
end

function apply(
    df::AbstractDataFrame,
    selector::MeasuresFilter{RankingLimiter}
)::Vector{Integer}
    mrlock = ReentrantLock()
    nbest = nbest(limiter(selector))
    n_cols = ncol(df)

    nbest >= n_cols && return trues(n_cols) # return immediately if 'k' is greater than columns number

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
    uselector = usedselector(selector)
    ranks = fill(0, n_cols)
    for mdf in measures_df
        ranks .+= buildbitmask(mdf, uselector)
    end

    return apply_limiter(vars, limiter(selector))
end
