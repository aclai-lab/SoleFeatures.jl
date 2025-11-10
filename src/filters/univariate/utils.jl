# ========================================================================================
# COMPOUND STATISTICAL FILTER
# ========================================================================================

# TODO: implement versus mode
struct CompoundStatisticalFilter{T<:AbstractLimiter} <: AbstractStatisticalFilter{T}
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
    ) where {T<:AbstractLimiter}
        !(versus in [:ovo, :ova]) && throw(ErrorException("Versus not allowed.\nValid symbols:\n\t:ova = one vs all\n:ovo = one vs one "))
        return new{T}(limiter, paramtest, nonparamtest, normalitycheck, versus, verbose)
    end

    function CompoundStatisticalFilter(
        limiter::T,
        paramtest::Any, # HypothesisTests.HypothesisTest,
        nonparamtest::Any, # HypothesisTests.HypothesisTest,
        versus::Symbol,
        verbose::Bool
    ) where {T<:AbstractLimiter}
        return new{T}(limiter, paramtest, nonparamtest, is_normal_distribuited, versus, verbose)
    end

    function CompoundStatisticalFilter(
        limiter::T,
        paramtest::Any, # HypothesisTests.HypothesisTest,
        nonparamtest::Any # HypothesisTests.HypothesisTest
    ) where {T<:AbstractLimiter}
        return new{T}(limiter, paramtest, nonparamtest, is_normal_distribuited, :ova, true)
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
# TRAITS

is_supervised(::CompoundStatisticalFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:SoleData.SoleBase.CLabel},
    selector::CompoundStatisticalFilter
)::DataFrame
    vrs = versus(selector)
    v = verbose(selector)
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
            useparamtest = all(normalitycheck(selector), [s1, s2])
            if (useparamtest)
                v && @info "Using param test on variable $(names(X)[cidx]), $(classes[c]) vs $(classes[vs])"
                stattest = paramtest(selector)
            else
                v && @info "Using non param test on variable $(names(X)[cidx]), $(classes[c]) vs $(classes[vs])"
                stattest = nonparamtest(selector)
            end
            push!(pvals, HypothesisTests.pvalue(stattest(s1, s2)))
        end
        push!(scores, pvals)
    end
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

function CompoundStatisticalMajority(
    paramtest::Any, # HypothesisTests.HypothesisTest,
    nonparamtest::Any; # HypothesisTests.HypothesisTest;
    normalitycheck::Function = is_normal_distribuited,
    versus::Symbol = :ova,
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
    versus::Symbol = :ova,
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

# ========================================================================================
# VARIANCE DISTANCE FILTER
# ========================================================================================

abstract type AbstractVarianceDistanceFilter{T<:AbstractLimiter} <: AbstractUnivariateFilterBased{T} end

struct VarianceDistanceFilter{T<:AbstractLimiter} <: AbstractVarianceDistanceFilter{T}
    limiter::T
    # parameters
    versus::Symbol

    function VarianceDistanceFilter(limiter::T, versus::Symbol) where {T<:AbstractLimiter}
        !(versus in [:ovo, :ova]) && throw(ErrorException("Versus not allowed.\nValid symbols:\n\t:ova = one vs all\n:ovo = one vs one "))
        return new{T}(limiter, versus)
    end
end

# ========================================================================================
# ACCESSORS

versus(s::VarianceDistanceFilter) = s.versus

# ========================================================================================
# TRAITS

is_supervised(::VarianceDistanceFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:SoleData.SoleBase.CLabel},
    selector::VarianceDistanceFilter
)
    vrs = versus(selector)

    numcol = size(X, 2)
    groupx = _group_by_class(X, y)
    # extract and remove classes from groupx
    classes = groupx[:, :class]
    select!(groupx, Not(:class))
    nclass = length(classes)
    ic = 1:nclass

    # orgiginal variance of each column
    ovars = StatsBase.var.(eachcol(X))
    # Each element is an array with 2 elements.
    # The first item is the index of the class considered and
    # the second the index, or indices, of the classes with which the first item is compared
    itr = Vector(vrs == :ovo ?
            collect(subsets(ic, 2)) :
            [ [first(setdiff(ic, x)), x] for x in subsets(ic, nclass - 1) ]
        )
    scores = Vector{Float64}(undef, numcol)
    for cidx in 1:numcol
        score = -Inf
        for (c, vs) in itr
            # retrive versus classes instances and compute variance on them
            svar = StatsBase.var(vcat(groupx[vs, cidx]...))
            # retrive current class instances and compute variance on it
            cvar = StatsBase.var(groupx[c, cidx])
            score = maximum([score, minimum(ovars[cidx] .- [svar, cvar])])
        end
        scores[cidx] = score
    end
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

function VarianceDistanceRanking(nbest::Integer; versus::Symbol = :ova)
    return VarianceDistanceFilter(RankingLimiter(nbest, true), versus)
end
