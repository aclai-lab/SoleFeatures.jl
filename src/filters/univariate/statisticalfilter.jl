struct StatisticalFilter{T <: AbstractLimiter} <: AbstractStatisticalFilter{T}
    limiter::T
    # parameters
    htest::Any # HypothesisTests.HypothesisTest
    versus::Symbol

    function StatisticalFilter(
        limiter::T,
        htest::Any, # HypothesisTests.HypothesisTest,
        versus::Symbol
    ) where {T <: AbstractLimiter}
        if (!(versus in VERSUS_SYMBOLS))
            throw(DomainError("Not valid `versus` symbol.\nAllowed symbols: $(VERSUS_SYMBOLS)"))
        end
        return new{T}(limiter, htest, versus)
    end
end

# ========================================================================================
# CONSTS

const VERSUS_SYMBOLS = [:ovo, :ova]

# ========================================================================================
# ACCESSORS

htest(selector::StatisticalFilter) = selector.htest
versus(selector::StatisticalFilter) = selector.versus

# ========================================================================================
# TRAITS

is_supervised(::AbstractStatisticalFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::StatisticalFilter
)::DataFrame
    stattest = htest(selector)
    vs = versus(selector)

    gdf = _group_by_class(X, y)
    classes = gdf[:, :class]
    attrs = names(X)
    scores = DataFrame()

    for (c1, c2) in IterTools.subsets(classes, 2)
        # class indices row in gdf
        c1_idx = findfirst(==(c1), classes)
        c2_idx = findfirst(==(c2), classes)
        pvals = []
        for attr in attrs
            # get and clear samples (clear or not to clear data?)
            s1 = filter(!isnan, gdf[c1_idx, attr])
            s2 = filter(!isnan, gdf[c2_idx, attr])
            push!(pvals, HypothesisTests.pvalue(stattest(s1, s2)))
        end
        insertcols!(scores, "$(c1)-vs-$(c2)" => pvals)
    end
    return scores
end

# ========================================================================================
# CUSTOM LIMITER

struct StatisticalLimiter{T<:AbstractLimiter} <: AbstractLimiter
    limiter::T
end

function limit(scores::DataFrame, sl::StatisticalLimiter)
    return limit(collect.(collect(eachrow(scores))), sl.limiter)
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

function StatisticalMajority(htest::Any; versus=:ovo) # HypothesisTests.HypothesisTest; versus=:ovo)
    sl = StatisticalLimiter(MajorityLimiter(ThresholdLimiter(0.05, <=)))
    return StatisticalFilter(sl, htest, versus)
end

function StatisticalAtLeastOnce(htest::Any; versus=:ovo) # HypothesisTests.HypothesisTest; versus=:ovo)
    sl = StatisticalLimiter(AtLeastLimiter(ThresholdLimiter(0.05, <=), 1))
    return StatisticalFilter(sl, htest, versus)
end
