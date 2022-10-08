"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
function minmax_normalize(
    df::AbstractDataFrame;
    min_quantile::Float64=0.0,
    max_quantile::Float64=1.0
)::DataFrame
    if min_quantile < 0.0
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    end
    if max_quantile > 1.0
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    end
    if max_quantile <= min_quantile
        throw(ErrorException("max_quantile must be greater then min_quantile"))
    end

    norm_df = DataFrame()

    for col_name in names(df)
        col = df[:, Symbol(col_name)]
        flatted_col = collect(Iterators.flatten(col))
        dim = SoleBase.dimension(DataFrame(:curr => col))
        tmin = StatsBase.quantile(flatted_col, min_quantile)
        tmax = StatsBase.quantile(flatted_col, max_quantile)
        tmax = 1 / (tmax - tmin)
        dt = StatsBase.UnitRangeTransform(1, 1, true, [tmin], [tmax])

        if dim == 0
            norm_col = StatsBase.transform(dt, Float64.(col))
        elseif dim == 1
            norm_col = map(r->StatsBase.transform(dt, Float64.(r)),
                Iterators.flatten(eachrow(col)))
        else
            error("unimplemented for dimension >2")
        end

        insertcols!(norm_df, Symbol(col_name) => norm_col)
    end

    return norm_df
end

function minmax_normalize(
    mfd::SoleBase.MultiFrameDataset,
    frame_index::Integer;
    min_quantile::Float64=0.0,
    max_quantile::Float64=1.0
)
    ndf = DataFrame()
    df = SoleBase.SoleDataset.data(mfd)
    attr_names = names(df)
    frames_descriptor = SoleBase.SoleDataset.frame_descriptor(mfd)
    frame_indices = frames_descriptor[frame_index]
    frame = SoleBase.frame(mfd, frame_index)
    norm_frame = minmax_normalize(frame; min_quantile=min_quantile, max_quantile=max_quantile)

    frame_i = 1
    for (i, name) in enumerate(attr_names)
        if (i in frame_indices)
            col = norm_frame[:,frame_i]
            frame_i += 1
        else
            col = df[:,i]
        end
        insertcols!(ndf, Symbol(name) => col)
    end

    return MultiFrameDataset(frames_descriptor, ndf)
end

"""
    _fr_bm2mfd_bm(mfd, frame_index, frame_bm)

frame bitmask to MultiFrameDataset bitmask.

return bitmask for entire MultiFrameDataset from a frame of it
"""
function _fr_bm2mfd_bm(
    mfd::SoleBase.MultiFrameDataset,
    frameidxes::Union{Integer, AbstractVector{<:Integer}},
    framebms::Union{BitVector, AbstractVector{<:BitVector}}
)::BitVector
    frameidxes = [ frameidxes... ]
    isa(framebms, BitVector) && (framebms = [ framebms ])

    length(frameidxes) != length(framebms) && throw(DimensionMismatch(""))

    bm = trues(nattributes(mfd))
    for i in 1:length(frameidxes)
        fridx = frameidxes[i]
        frbm = framebms[i]
        framedescr = SoleBase.SoleDataset.frame_descriptor(mfd)[fridx] # frame indices inside mfd
        bm[framedescr] = frbm
    end
    return bm
end

"""
    bm2attr

return tuple containing names of suitable attributes and names of not suitable attributes
"""
function bm2attr(mfd::SoleBase.MultiFrameDataset, bm::BitVector)
    return bm2attr(SoleBase.SoleData.SoleDataset.data(mfd), bm)
end

function bm2attr(mfd::SoleBase.MultiFrameDataset, fridx::Integer, bm::BitVector)
    return bm2attr(SoleBase.frame(mfd, fridx), bm)
end

function bm2attr(df::AbstractDataFrame, bm::BitVector)
    attr = names(df)
    good_attr = attr[findall(bm)]
    bad_attr = attr[findall(!, bm)]
    return good_attr, bad_attr
end

function _dropattr_fromframe!(
    mfd::AbstractMultiFrameDataset,
    frmidx::Integer,
    frmattridx::Union{Integer, AbstractVector{<:Integer}}
)
    frmattridx = [ frmattridx... ]
    nattrfrm = nattributes(mfd, frmidx)

    !(1 <= frmidx <= nframes(mfd)) && throw(DimensionMismatch("frmidx"))

    attridx = SoleBase.SoleDataset.frame_descriptor(mfd)[frmidx][frmattridx]

    return SoleBase.dropattributes!(mfd, attridx)
end
