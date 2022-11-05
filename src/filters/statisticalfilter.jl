struct StatisticalFilter{T <: AbstractFilterLimiter} <: AbstractStatisticalFilter{T}
    limiter::T
    # parameters
    param_tests::Vector
    non_param_tests::Vector

    function StatisticalFilter(
        limiter::T,
        param_tests::Vector,
        non_param_tests::Vector
    ) where {T <: AbstractFilterLimiter}
        return new{T}(limiter, param_tests, non_param_tests)
    end

    function StatisticalFilter(limiter::T) where {T <: AbstractFilterLimiter}
        return StatisticalFilter(
            limiter,
            [
                HypothesisTests.EqualVarianceZTest,
                HypothesisTests.UnequalVarianceZTest,
                HypothesisTests.EqualVarianceTTest,
                HypothesisTests.UnequalVarianceTTest
            ],
            [
                HypothesisTests.MannWhitneyUTest,
                HypothesisTests.ApproximateTwoSampleKSTest,
                # HypothesisTests.SignedRankTest
            ]
        )
    end
end

limiter(selector::StatisticalFilter) = selector.limiter
param_tests(selector::StatisticalFilter) = selector.param_tests
non_param_tests(selector::StatisticalFilter) = selector.non_param_tests

is_supervised(::AbstractStatisticalFilter{<:AbstractFilterLimiter}) = true

# ========================================================================================
# Constructors

# Threshold

function StatisticalThreshold(threshold::Real)
    return StatisticalFilter(ThresholdLimiter(threshold, >=))
end

# ========================================================================================
# Shared apply functions

function apply(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::StatisticalFilter{<:AbstractFilterLimiter}
)
    !is_supervised(selector) && throw(ErrorException("Only supervised selector allowed"))
    gdf = _group_by_class(X, y)
    classes = gdf[:, :class]
    attrs = names(X)

    scores = Vector{Int}() # times whose attribute was significatn for a class
    for attr in attrs
        curr_passed = []
        for (c1, c2) in IterTools.subsets(classes, 2)
            # class indices row in gdf
            c1_idx = findfirst(==(c1), classes)
            c2_idx = findfirst(==(c2), classes)
            # get and clear samples (clear or not to clear data?)
            s1 = filter(!isnan, gdf[c1_idx, attr])
            s2 = filter(!isnan, gdf[c2_idx, attr])
            # check samples normality
            useparamtest = all(is_normal_distribuited, [s1, s2])
            stattests = useparamtest ? param_tests(selector) : non_param_tests(selector)
            # apply param tests
            pvals = _statistical_test(s1, s2, stattests)
            # TODO: parameterize passed constraint: at least one passed, half passed, all passed
            passed = length(findall(<=(0.05), pvals)) >= (length(stattests) * 0.5)
            push!(curr_passed, passed)
        end
        push!(scores, sum(curr_passed))
    end
    return apply_limiter(scores, limiter(selector))
end

"""
    is_normal_distribuited(population)

Return true if population is normally distribuited, false otherwise

## Info

Tecnique comes from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/
"""
function is_normal_distribuited(population::AbstractVector{<:Real})
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/
    # 1. background for dimension upper 40 (central limit theorem)
    # 5. conclusion for Shapiro-Wilk test
    return length(population) > 40 || normality_shapiro(population) > 0.05
end

"""
Perform Shapiro-Wilk normality test on a population.
"""
function normality_shapiro(population::AbstractVector{<:Real})
    isempty(population) && throw(DimensionMismatch("empty population"))
    stat, p = pyimport("scipy.stats").shapiro(population)
    return p
end

"""
    _statistical_test(s1, s2, stattests)

Perform indicated statistical tests (`stattests`) on `s1` and `s2`.

Return pvalues of each `stattests`
"""
function _statistical_test(
    s1::AbstractVector,
    s2::AbstractVector,
    stattests::AbstractVector;
)
    return HypothesisTests.pvalue.([ t(s1, s2) for t in stattests ])
end
