"""
Perform provided hypothesis test `htest` (from: https://github.com/JuliaStats/HypothesisTests.jl)
on each attributes.
The attribute is splitted into as many populations as there are pairs of classes (n*(n-1)/2 pairs class_i vs class_j)
and the hypothesis test is performed between two population of different classes.
"""
struct StatisticalFilter{T <: AbstractLimiter} <: AbstractStatisticalFilter{T}
    limiter::T
    # parameters
    htest::Any # HypothesisTests.HypothesisTest
end

# ========================================================================================
# ACCESSORS

htest(selector::StatisticalFilter) = selector.htest

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

function StatisticalMajority(htest::Any; significance = 0.05, rejectnullhp = true) # HypothesisTests.HypothesisTest; versus=:ovo)
    rejectnull = rejectnullhp ? (<=) : (>)
    sl = StatisticalLimiter(MajorityLimiter(ThresholdLimiter(significance, rejectnull)))
    return StatisticalFilter(sl, htest)
end

function StatisticalAtLeastOnce(htest::Any; significance = 0.05, rejectnullhp = true) # HypothesisTests.HypothesisTest; versus=:ovo)
    rejectnull = rejectnullhp ? (<=) : (>)
    sl = StatisticalLimiter(AtLeastLimiter(ThresholdLimiter(significance, rejectnull), 1))
    return StatisticalFilter(sl, htest)
end
