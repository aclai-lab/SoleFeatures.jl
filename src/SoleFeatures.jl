# TODO: make comments in minmax_normalize
# TODO: utils.minmax_normalize shouldn't flat cols
# TODO: utils.minmax_normalize test dimension 2
# TODO: better implmentation of selector_function on correlation_ranking and correlation_threshold

module SoleFeatures

using DataFrames
using SoleBase
using StatsBase
using DynamicAxisWarping

# -----------------------------------------------------------------------------------------
# exports

export VarianceThreshold
export VarianceRanking
export CorrelationThreshold
export CorrelationRanking

# -----------------------------------------------------------------------------------------
# abstract types

"""
Abstract supertype for all features selector.

A concrete subtype of AbstractFeaturesSelector should always provide functions
[`apply`](@ref) and [`build_bit_mask`](@ref)
"""
abstract type AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractFilterBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractWrapperBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstractEmbeddedBased <: AbstractFeaturesSelector end


# -----------------------------------------------------------------------------------------
# AbstractFeaturesSelector

"""
    apply(mfd, selector)

Return a new MultiFrameDataset from `mfd` without the attributes considered unsitable from `selector`
## ARGUMENTS
- `mfd::AbstractMultiFrameDataset`: AbstractMultiFrameDataset on which apply the selector
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    normalize_function=nothing
)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_index::Int;
    normalize_function=nothing
)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

"""
    apply!(mfd, selector)

Remove form `mfd` attributes considered unsitable from `selector`

## ARGUMENTS
- `mfd::AbstractMultiFrameDataset`: AbstractMultiFrameDataset on which apply the selector
- `selector::AbstractFeaturesSelector`: applied selector
"""
function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    normalize_function=nothing
)
    return error("`apply!` not implmented for type: "
        * string(typeof(selector)))
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_index::Int;
    normalize_function=nothing
)
    return error("`apply!` not implmented for type: "
        * string(typeof(selector)))
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    return error("`apply!` not implmented for type: "
        * string(typeof(selector)))
end

"""
    build_bit_mask(df, selector)

Return a bit vector representing which attributes in `df` are considered suitable or not by the `selector`
(1 suitable, 0 not suitable)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame to evaluate
- `selector::AbstractFeaturesSelector`: applied selector
"""
function build_bitmask(df::AbstractDataFrame, selector::AbstractFeaturesSelector)::BitVector
    return error("`build_bit_mask` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# AbstractFilterBased - threshold

function selector_threshold(selector::AbstractFilterBased)
    return error("`selector_threshold` not implmented for type: "
        * string(typeof(selector)))
end

function selector_function(selector::AbstractFilterBased)
    return error("`selector_function` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# AbstractFilterBased - ranking

function selector_k(selector::AbstractFilterBased)
    return error("`selector_k` not implmented for type: "
        * string(typeof(selector)))
end

function selector_rankfunct(selector::AbstractFilterBased)
    return error("`selector_rankfunct` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# DTW AVG Correlation

"""
    _compute_dtw(df)

Compute DTW between each timeseries in a column for each column of `df`

Returns a matrix of nr*(nr-1)/2 rows (with nr number of rows in `df`) and nc columns (with nc number of columns in `df`)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate DTW
"""
function _compute_dtw(df::AbstractDataFrame)::Array{Float64,2}
    # maybe a better implmentation do dtw only on a vector of timeseries

    nr, nc = size(df)
    # number of rows in the result matrix
    nrm = Int((nr*(nr-1))/2)
    # distances matrix
    dist_matrix = Array{Float64, 2}(undef, nrm, nc)
    # computation of the dtw for each timeseries in a column for each attribute in df
    Threads.@threads for cidx in 1:nc
        idxm = 1
        for iidx in 1:(nr-1)
            for jidx in (iidx+1):nr
                # dtw returns cost and a set of indices (i1,i2) that align the two serie, so only cost (dtw(...)[1])
                # have to be extracted
                dist_matrix[idxm, cidx] = dtw(df[iidx, cidx], df[jidx, cidx])[1]
                idxm = idxm + 1
            end
        end
    end
    return dist_matrix
end

"""
    dtw_correlation(df, corf)

Returns mean absolute correlation vector, based on dtw

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame on which to calculate mean absolute correlation vector
- `corf::Function`: correlation function, function that generates the correlation matrix
"""
function dtw_correlation(df::AbstractDataFrame, corf::Function)::Array{Float64}
    # build distances matrix
    dist_matrix = _compute_dtw(df)
    # correlation matrix built from the correlation function provided (corf)
    # absolute value of each coefficient is calculated
    cor_matrix = abs.(corf(dist_matrix))
    # NaN values obtained from equal time series are converted into 1
    replace!(cor_matrix, NaN=>1)
    # calculate avg of correlation matrix per column (mean absolute correlation vector)
    avg_vector = vec(mean(cor_matrix, dims=1))
    return avg_vector
end

include("./utils.jl")
include("./variance_threshold.jl")
include("./variance_ranking.jl")
include("./correlation_threshold.jl")
include("./correlation_ranking.jl")

end # module
