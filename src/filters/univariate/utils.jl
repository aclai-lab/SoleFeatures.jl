# ========================================================================================
# COMPOUND STATISTICAL FILTER
# ========================================================================================

struct CompoundStatisticalFilter{T <: AbstractLimiter} <: AbstractStatisticalFilter{T}
    limiter::T
    # parameters
    paramtest::Any # HypothesisTests.HypothesisTest
    nonparamtest::Any # HypothesisTests.HypothesisTest
    normalitycheck::Function
    versus::Symbol
    verbose::Bool

    function CompoundStatisticalFilter(
        limiter::T,
        paramtest::Any, # HypothesisTests.HypothesisTest,
        nonparamtest::Any, # HypothesisTests.HypothesisTest,
        normalitycheck::Function,
        versus::Symbol,
        verbose::Bool
    ) where {T <: AbstractLimiter}
        if (!(versus in VERSUS_SYMBOLS))
            throw(DomainError("Not valid `versus` symbol.\nAllowed symbols: $(VERSUS_SYMBOLS)"))
        end
        return new{T}(limiter, paramtest, nonparamtest, normalitycheck, versus, verbose)
    end

    function CompoundStatisticalFilter(
        limiter::T,
        paramtest::Any, # HypothesisTests.HypothesisTest,
        nonparamtest::Any, # HypothesisTests.HypothesisTest,
        verbose::Bool
    ) where {T <: AbstractLimiter}
        if (!(versus in VERSUS_SYMBOLS))
            throw(DomainError("Not valid `versus` symbol.\nAllowed symbols: $(VERSUS_SYMBOLS)"))
        end
        return new{T}(limiter, paramtest, nonparamtest, is_normal_distribuited, :ovo, verbose)
    end

    function CompoundStatisticalFilter(
        limiter::T,
        paramtest::Any, # HypothesisTests.HypothesisTest,
        nonparamtest::Any # HypothesisTests.HypothesisTest
    ) where {T <: AbstractLimiter}
        if (!(versus in VERSUS_SYMBOLS))
            throw(DomainError("Not valid `versus` symbol.\nAllowed symbols: $(VERSUS_SYMBOLS)"))
        end
        return new{T}(limiter, paramtest, nonparamtest, is_normal_distribuited, :ovo, true)
    end
end

# ========================================================================================
# ACCESSORS

paramtest(selector::CompoundStatisticalFilter) = selector.paramtest
nonparamtest(selector::CompoundStatisticalFilter) = selector.nonparamtest
normalitycheck(selector::CompoundStatisticalFilter) = selector.normalitycheck
versus(selector::CompoundStatisticalFilter) = selector.versus
verbose(selector::CompoundStatisticalFilter) = selector.verbose

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Union{String, Symbol}},
    selector::CompoundStatisticalFilter
)::DataFrame
    v = verbose(selector)
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
            useparamtest = all(normalitycheck(selector), [s1, s2])
            if (useparamtest)
                v && @info "Using param test on attribute $(attr), $(c1) vs $(c2)"
                stattest = paramtest(selector)
            else
                v && @info "Using non param test on attribute $(attr), $(c1) vs $(c2)"
                stattest = nonparamtest(selector)
            end
            push!(pvals, HypothesisTests.pvalue(stattest(s1, s2)))
        end
        insertcols!(scores, "$(c1)-vs-$(c2)" => pvals)
    end
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

function CompoundStatisticalMajority(
    paramtest::Any, # HypothesisTests.HypothesisTest,
    nonparamtest::Any; # HypothesisTests.HypothesisTest;
    normalitycheck::Function = is_normal_distribuited,
    versus::Symbol = :ovo,
    verbose::Bool = false
)
    sl = StatisticalLimiter(MajorityLimiter(ThresholdLimiter(0.05, <=)))
    return CompoundStatisticalFilter(
        sl,
        paramtest,
        nonparamtest,
        normalitycheck,
        versus,
        verbose
    )
end

function CompoundStatisticalAtLeastOnce(
    paramtest::Any, # HypothesisTests.HypothesisTest,
    nonparamtest::Any; # HypothesisTests.HypothesisTest;
    normalitycheck::Function = is_normal_distribuited,
    versus::Symbol = :ovo,
    verbose::Bool = false
)
    sl = StatisticalLimiter(AtLeastLimiter(ThresholdLimiter(0.05, <=), 1))
    return CompoundStatisticalFilter(
        sl,
        paramtest,
        nonparamtest,
        normalitycheck,
        versus,
        verbose
    )
end

# ========================================================================================
# UTILS

"""
    is_normal_distribuited(population)

Return true if population is normally distribuited, false otherwise

## Info

Tecnique comes from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/
"""
function is_normal_distribuited(population::AbstractVector{<:Real})
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3693611/
    # 1. background for dimension upper 40 (central limit theorem)
    # 5. conclusion for Shapiro-Wilk test (null hypothesis is that "sample distribution is normal")
    return length(population) > 40 || normality_shapiro(population) > 0.05
end

"""
Perform Shapiro-Wilk normality test on a population.
"""
function normality_shapiro(population::AbstractVector{<:Real})
    isempty(population) && throw(DimensionMismatch("empty population"))
    _, p = pyimport("scipy.stats").shapiro(population)
    return p
end
