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

# function apply(df::AbstractDataFrame, bm::BitVector)
#     TODO: maybe a day it will implmented
#     indices = findall(==(true), bm)
#     return select(df, indices)
# end

function apply!(mfd::SoleBase.AbstractMultiFrameDataset, bm::BitVector)
    indices = findall(==(false), bm)
    return SoleBase.SoleDataset.dropattributes!(mfd, indices)
end

function apply!(mfd::SoleBase.AbstractMultiFrameDataset, bms::AbstractVector{BitVector})
    nframe(mfd) != length(bms) && throw(DimensionMismatch(""))
    bm = reduce(vcat, bms) # concat bitmask vectors in one vector
    return apply!(mfd, bm)
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    normalize_function=nothing
)
    df = SoleBase.SoleDataset.data(mfd)
    @assert all(col -> (col isa Union{Array{<:Number},Number}),
        collect(Iterators.flatten(eachcol(df))))
    "Attributes are not numerical type"

    if !isnothing(normalize_function)
        df_norm = normalize_function(df)
        bm = build_bitmask(df_norm, selector)
    else
        bm = build_bitmask(df, selector)
    end

    return apply!(mfd, bm)
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::Union{Integer, Array{Integer}};
    normalize_function=nothing
)
    frame_indices = !isa(frame_indices, Array) ? [frame_indices] : frame_indices
    for idx in frame_indices
        # frame from 'frame_index'
        fr = SoleBase.frame(mfd, idx)
        @assert all(col->(col isa Union{Array{<:Number},Number}),
                    collect(Iterators.flatten(eachcol(fr))))
                        "Attributes are not numerical type"

        # check if the frame needs normalization
        if !isnothing(normalize_function)
            fr_norm = normalize_function(fr)
            fr_bm = build_bitmask(fr_norm, selector)
        else
            fr_bm = build_bitmask(fr, selector)
        end

        # bit mask for entire dataset
        bm = trues(nattributes(mfd))
        fr_indices = SoleBase.SoleDataset.frame_descriptor(mfd)[idx] # frame indices inside mfd
        bm[fr_indices] = fr_bm

        apply!(mfd, bm)
    end

    return mfd
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::Union{Integer, Array{Integer}};
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector, frame_indices; normalize_function=normalize_function)
    return mfd_clone
end

"""
    build_bit_mask(df, selector)

Return a bit vector representing which attributes in `df` are considered suitable or not by the `selector`
(1 suitable, 0 not suitable)

## ARGUMENTS
- `df::AbstractDataFrame`: DataFrame to evaluate
- `selector::AbstractFeaturesSelector`: applied selector
"""
function build_bitmask(
    df::AbstractDataFrame,
    selector::AbstractFeaturesSelector
)::BitVector
    return error("`build_bitmask` not implmented for type: "
                 *
                 string(typeof(selector)))
end

function build_bitmask(
    mfd::SoleBase.MultiFrameDataset,
    frame_index::Union{Integer, Array{Integer}},
    selector::AbstractFeaturesSelector
)::BitVector

end

# -----------------------------------------------------------------------------------------
# AbstractFilterBased - threshold

function selector_threshold(selector::AbstractFilterBased)
    return error("`selector_threshold` not implmented for type: "
                 *
                 string(typeof(selector)))
end

function selector_function(selector::AbstractFilterBased)
    return error("`selector_function` not implmented for type: "
                 *
                 string(typeof(selector)))
end

# -----------------------------------------------------------------------------------------
# AbstractFilterBased - ranking

function selector_k(selector::AbstractFilterBased)
    return error("`selector_k` not implmented for type: "
                 *
                 string(typeof(selector)))
end

function selector_rankfunct(selector::AbstractFilterBased)
    return error("`selector_rankfunct` not implmented for type: "
                 *
                 string(typeof(selector)))
end
