module SoleFeatures

using DataFrames
using SoleBase
using StatsBase

# -----------------------------------------------------------------------------------------
# exports

export VarianceThreshold
export VarianceRanking

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
abstract type AbstarctFilterBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstarctWrapperBased <: AbstractFeaturesSelector end

"""
Abstract supertype filter based selector.
"""
abstract type AbstarctEmbeddedBased <: AbstractFeaturesSelector end


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
    frame_index::Integer;
    normalize_function=nothing
)
    return error("`apply` not implmented for type: "
        * string(typeof(selector)))
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::AbstractVector{<:Integer};
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
    frame_index::Integer;
    normalize_function=nothing
)
    return error("`apply!` not implmented for type: "
        * string(typeof(selector)))
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::AbstractVector{<:Integer};
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
# AbstarctFilterBased - threshold

function selector_threshold(selector::AbstarctFilterBased)
    return error("`selector_threshold` not implmented for type: "
        * string(typeof(selector)))
end

function selector_function(selector::AbstarctFilterBased)
    return error("`selector_function` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# AbstarctFilterBased - ranking

function selector_k(selector::AbstarctFilterBased)
    return error("`selector_k` not implmented for type: "
        * string(typeof(selector)))
end

function selector_rankfunct(selector::AbstarctFilterBased)
    return error("`selector_rankfunct` not implmented for type: "
        * string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# utils

"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
function minmax_normalize(df::AbstractDataFrame)::DataFrame
    norm_df = DataFrame()

    for col_name in names(df)
        col = df[:, Symbol(col_name)]
        flatted_col = collect(Iterators.flatten(col))
        dim = SoleBase.dimension(DataFrame(:curr => col))
        dt = fit(UnitRangeTransform, Float64.(flatted_col), dims=1)
        
        if dim == 0
            norm_col = StatsBase.transform(dt, Float64.(col))
        elseif dim == 1
            norm_col = map(r->StatsBase.transform(dt, Float64.(r)),
                Iterators.flatten(eachrow(col)))
        else
            error("unimplemented for dimension >1")
        end

        insertcols!(norm_df, Symbol(col_name) => norm_col)
    end

    return norm_df
end

include("./VarianceThreshold.jl")
include("./VarianceRanking.jl")

end # module
