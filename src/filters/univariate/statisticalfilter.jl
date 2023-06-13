"""
Perform provided hypothesis test `htest` (from: https://github.com/JuliaStats/HypothesisTests.jl)
on each variables.
The variable is splitted into as many populations as there are pairs of classes (n*(n-1)/2 pairs class_i vs class_j)
and the hypothesis test is performed between two population of different classes.
"""
struct StatisticalFilter{T <: AbstractLimiter} <: AbstractStatisticalFilter{T}
    limiter::T
    # parameters
    htest::Any # HypothesisTests.HypothesisTest
    versus::Symbol
end

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
    y::AbstractVector{<:Class},
    selector::StatisticalFilter
)::DataFrame
    stattest = htest(selector)
    vrs = versus(selector)
    numcol = size(X, 2)
    groupx = _group_by_class(X, y)
    # extract and remove classes from groupx
    classes = groupx[:, :class]
    select!(groupx, Not(:class))
    nclass = length(classes)
    ic = 1:nclass

    # Each element is an array with 2 elements.
    # The first item is the index of the class considered and
    # the second the index, or indices, of the classes with which the first item is compared
    itr = Vector(vrs == :ovo ?
            collect(subsets(ic, 2)) :
            [ [first(setdiff(ic, x)), x] for x in subsets(ic, nclass - 1) ]
    )

    colnames = join.([ [classes[c], classes[vs]] for (c, vs) in itr ], "-vs-")
    scores = DataFrame(colnames .=> [Float64[]])
    for cidx in 1:numcol
        pvals = []
        for (c, vs) in itr
            s1 = groupx[c, cidx]
            s2 = vcat(groupx[vs, cidx]...)
            # clear samples from nan
            filter!(!isnan, s1)
            filter!(!isnan, s2)
            push!(pvals, HypothesisTests.pvalue(stattest(s1, s2)))
        end
        push!(scores, pvals)
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

function StatisticalMajority(
    htest::Any;
    versus::Symbol = :ova,
    significance::Real = 0.05,
    rejectnullhp = true
)
    (significance < 0 || significance > 1) &&
        throw(DomainError("significance must be within 0 and 1"))
    rejectnull = rejectnullhp ? (<=) : (>)
    sl = StatisticalLimiter(MajorityLimiter(ThresholdLimiter(significance, rejectnull)))
    return StatisticalFilter(sl, htest, versus)
end

function StatisticalAtLeastOnce(
    htest::Any;
    versus = :ova,
    significance = 0.05,
    rejectnullhp = true
)
    (significance < 0 || significance > 1) &&
        throw(DomainError("significance must be within 0 and 1"))
    rejectnull = rejectnullhp ? (<=) : (>)
    sl = StatisticalLimiter(AtLeastLimiter(ThresholdLimiter(significance, rejectnull), 1))
    return StatisticalFilter(sl, htest, versus)
end
