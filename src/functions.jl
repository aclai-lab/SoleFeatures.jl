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

        bm = _fr_bm2mfd_bm(mfd, idx, fr_bm)

        apply!(mfd, bm)
    end

    return mfd
end

function apply(mfd::SoleBase.AbstractMultiFrameDataset, bm::BitVector)
    nmfd = deepcopy(mfd)
    apply!(nmfd, bm)
    return nmfd
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

function build_bitmask(
    mfd::SoleBase.MultiFrameDataset,
    frame_index::Integer,
    selector::AbstractFeaturesSelector
)::Tuple{BitVector, BitVector}
    fr = SoleBase.frame(mfd, frame_index)
    fr_bm = build_bitmask(fr, selector) # frame bitmask
    bm = _fr_bm2mfd_bm(mfd, frame_index, fr_bm)
    return bm, fr_bm
end
