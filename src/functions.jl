function apply!(df::AbstractDataFrame, bm::BitVector)
    ncol(df) != length(bm) && throw(DimensionMismatch(""))
    return select!(df, findall(bm))
end

function apply!(mfd::SoleBase.AbstractMultiFrameDataset, bm::BitVector)
    nattributes(mfd) != length(bm) && throw(DimensionMismatch(""))
    return SoleBase.SoleDataset.dropattributes!(mfd, findall(!, bm))
end

function apply!(mfd::SoleBase.AbstractMultiFrameDataset, bms::AbstractVector{BitVector})
    nframe(mfd) != length(bms) && throw(DimensionMismatch(""))
    return apply!(mfd, reduce(vcat, bms))
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector
)
    df = SoleBase.SoleDataset.data(mfd)
    bm = build_bitmask(df, selector)
    return apply!(mfd, bm)
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    fr_indices::Union{Integer, Array{Integer}}
)
    fr_indices = [fr_indices...]
    for idx in fr_indices
        fr = SoleBase.frame(mfd, idx)
        frbm = build_bitmask(fr, selector)
        bm = _fr_bm2mfd_bm(mfd, idx, frbm)
        apply!(mfd, bm)
    end
    return mfd
end

apply(df::AbstractDataFrame, bm::BitVector) = apply!(deepcopy(df), bm)

apply(mfd::SoleBase.AbstractMultiFrameDataset, bm::BitVector) = apply!(deepcopy(mfd), bm)

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector
)
    return apply!(deepcopy(mfd), selector)
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector,
    frame_indices::Union{Integer, Array{Integer}};
)
    return apply!(deepcopy(mfd), selector, frame_indices)
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
