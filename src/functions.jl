"""
    transform!(df, args..; kwargs...)
    transform!(mfd, args..; kwargs...)

Removes from provided DataFrame attributes indicated by bitmask or selector

# Arguments

- `df::AbstractDataFrame`: Dataset
- `mfd::MultiFrameDataset`: Dataset
- `bm::BitVector`: Vector of bit containing which attributes are suitable(1) or not(0)
- `selector::AbstractFeaturesSelector`: Selector

# Keywords
- `frmidx::Integer`: Frame index inside `mfd`
"""
function transform!(df::AbstractDataFrame, bm::BitVector)
    ncol(df) != length(bm) && throw(DimensionMismatch(""))
    return select!(df, findall(bm))
end

function transform!(df::AbstractDataFrame, selector::AbstractFeaturesSelector)
    return transform!(df, buildbitmask(df, selector))
end

function transform!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    bm::BitVector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        nattributes(mfd) != length(bm) && throw(DimensionMismatch(""))
        return SoleBase.SoleDataset.dropattributes!(mfd, findall(!, bm))
    else
        nattributes(mfd, frmidx) != length(bm) && thow(DimensionMismatch(""))
        # return dropattributes!(mfd, frmidx, findall(!, bm))
        return SoleBase.SoleDataset.dropattributes!(mfd, frmidx, findall(!, bm))
    end
end

function transform!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::AbstractFeaturesSelector;
    frmidx::Union{Integer, Nothing} = nothing
)
    if (isnothing(frmidx))
        bm = buildbitmask(SoleBase.SoleDataset.data(mfd), selector)
    else
        bm = buildbitmask(SoleBase.frame(mfd, frmidx), selector)
    end
    return transform!(mfd, bm; frmidx=frmidx)
end

transform(df::AbstractDataFrame, args...; kwargs...) = transform!(deepcopy(df), args...; kwargs...)
transform(mfd::SoleBase.AbstractMultiFrameDataset, args...; kwargs...) = transform!(deepcopy(mfd), args...; kwargs...)

function buildbitmask(
    mfd::SoleBase.MultiFrameDataset,
    frmidx::Integer,
    selector::AbstractFeaturesSelector
)::Tuple{BitVector, BitVector}
    frbm = buildbitmask(SoleBase.frame(mfd, frmidx), selector) # frame bitmasks
    bm = _fr_bm2mfd_bm(mfd, frmidx, frbm)
    return bm, frbm
end

function buildbitmask(
    df::AbstractDataFrame,
    selector::AbstractFeaturesSelector
)::BitVector
    idxes = apply(df, selector)
    return _idxes2bm(size(df, 2), idxes)
end

function _idxes2bm(bmlen::Integer, idxes::AbstractVector{<:Integer})
    bm = falses(bmlen)
    bm[idxes] .= true
    return bm
end
