using SoleData

"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
# function minmax_normalize(
#     df::AbstractDataFrame;
#     min_quantile::Float64=0.0,
#     max_quantile::Float64=1.0
# )::DataFrame
#     if min_quantile < 0.0
#         throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
#     end
#     if max_quantile > 1.0
#         throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
#     end
#     if max_quantile <= min_quantile
#         throw(ErrorException("max_quantile must be greater then min_quantile"))
#     end

#     norm_df = DataFrame()

#     for col_name in names(df)
#         col = df[:, Symbol(col_name)]
#         flatted_col = collect(Iterators.flatten(col))
#         dim = SoleBase.dimension(DataFrame(:curr => col))
#         tmin = StatsBase.quantile(flatted_col, min_quantile)
#         tmax = StatsBase.quantile(flatted_col, max_quantile)
#         tmax = 1 / (tmax - tmin)
#         dt = StatsBase.UnitRangeTransform(1, 1, true, [tmin], [tmax])

#         if dim == 0
#             norm_col = StatsBase.transform(dt, Float64.(col))
#         elseif dim == 1
#             norm_col = map(r->StatsBase.transform(dt, Float64.(r)),
#                 Iterators.flatten(eachrow(col)))
#         else
#             error("unimplemented for dimension >2")
#         end

#         insertcols!(norm_df, Symbol(col_name) => norm_col)
#     end

#     return norm_df
# end

# function minmax_normalize(
#     mfd::SoleData.MultiFrameDataset,
#     frame_index::Integer;
#     min_quantile::Float64=0.0,
#     max_quantile::Float64=1.0
# )
#     ndf = DataFrame()
#     df = SoleData.data(mfd)
#     attr_names = names(df)
#     frames_descriptor = SoleData.frame_descriptor(mfd)
#     frame_indices = frames_descriptor[frame_index]
#     frame = SoleBase.frame(mfd, frame_index)
#     norm_frame = minmax_normalize(frame; min_quantile=min_quantile, max_quantile=max_quantile)

#     frame_i = 1
#     for (i, name) in enumerate(attr_names)
#         if (i in frame_indices)
#             col = norm_frame[:,frame_i]
#             frame_i += 1
#         else
#             col = df[:,i]
#         end
#         insertcols!(ndf, Symbol(name) => col)
#     end

#     return MultiFrameDataset(frames_descriptor, ndf)
# end

# """
#     _fr_bm2mfd_bm(mfd, frame_index, frame_bm)

# frame bitmask to MultiFrameDataset bitmask.

# return bitmask for entire MultiFrameDataset from a frame of it
# """
# function _fr_bm2mfd_bm(
#     mfd::SoleData.MultiFrameDataset,
#     frameidxes::Union{Integer, AbstractVector{<:Integer}},
#     framebms::Union{BitVector, AbstractVector{<:BitVector}}
# )::BitVector
#     frameidxes = [ frameidxes... ]
#     isa(framebms, BitVector) && (framebms = [ framebms ])

#     length(frameidxes) != length(framebms) && throw(DimensionMismatch(""))

#     bm = trues(nattributes(mfd))
#     for i in 1:lastindex(frameidxes)
#         fridx = frameidxes[i]
#         frbm = framebms[i]
#         framedescr = SoleData.frame_descriptor(mfd)[fridx] # frame indices inside mfd
#         bm[framedescr] = frbm
#     end
#     return bm
# end

minmax_normalize(c, args...; kwars...) = minmax_normalize!(deepcopy(c), args...; kwars...)

function minmax_normalize!(
    mfd::SoleData.MultiFrameDataset,
    frame_index::Integer;
    min_quantile::AbstractFloat=0.0,
    max_quantile::AbstractFloat=1.0,
    col_quantile::Bool=true,
)
    return minmax_normalize!(
        SoleBase.frame(mfd, frame_index);
        min_quantile=min_quantile,
        max_quantile=max_quantile, col_quantile
    )
end

function minmax_normalize!(
    df::AbstractDataFrame;
    min_quantile::AbstractFloat=0.0,
    max_quantile::AbstractFloat=1.0,
    col_quantile::Bool=true,
)
    min_quantile < 0.0 &&
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    max_quantile > 1.0 &&
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    max_quantile <= min_quantile &&
        throw(DomainError("max_quantile must be greater then min_quantile"))

    icols = eachcol(df)

    !all(==(AbstractFloat), supertype.(eltype.(icols))) &&
        throw(DomainError("DataFrame contains columns with type different from Float"))

    if (!col_quantile)
        # look for quantile in entire dataset
        itdf = Iterators.flatten(Iterators.flatten(icols))
        min = StatsBase.quantile(itdf, min_quantile)
        max = StatsBase.quantile(itdf, max_quantile)
    else
        # quantile for each column
        itcol = Iterators.flatten.(icols)
        min = StatsBase.quantile.(itcol, min_quantile)
        max = StatsBase.quantile.(itcol, max_quantile)
    end
    minmax_normalize!.(icols, min, max)
    return df
end

function minmax_normalize!(
    v::AbstractArray{<:AbstractArray{<:AbstractFloat}},
    min::Real,
    max::Real
)
    return minmax_normalize!.(v, min, max)
    # @Threads.threads for (i, iv) in collect(enumerate(v))
    #     v[i] = minmax_normalize(iv, min, max)
    # end
end

function minmax_normalize!(
    v::AbstractArray{<:AbstractFloat},
    min::Real,
    max::Real
)
    min = float(min)
    max = float(max)
    max = 1 / (max - min)
    rt = StatsBase.UnitRangeTransform(1, 1, true, [min], [max])
    # This function doesn't accept Integer
    return StatsBase.transform!(rt, v)
end

"""
    bm2attr

return tuple containing names of suitable attributes and names of not suitable attributes
"""
function bm2attr(mfd::SoleData.MultiFrameDataset, bm::BitVector)
    return bm2attr(SoleData.data(mfd), bm)
end

function bm2attr(mfd::SoleData.MultiFrameDataset, fridx::Integer, bm::BitVector)
    return bm2attr(SoleBase.frame(mfd, fridx), bm)
end

function bm2attr(df::AbstractDataFrame, bm::BitVector)
    attr = names(df)
    good_attr = attr[findall(bm)]
    bad_attr = attr[findall(!, bm)]
    return good_attr, bad_attr
end


"""
    _group_by_class(df, y)

Group a data frame by its classes.
Target column will be called "class" and it will be the last column of dataframe

# Examples

```julia-repl
julia> df = DataFrame(:firstcol => [1,2,3,4], :secondcol => [8,9,7,10])
4×2 DataFrame
 Row │ firstcol  secondcol
     │ Int64     Int64
─────┼─────────────────────
   1 │        1          8
   2 │        2          9
   3 │        3          7
   4 │        4         10

julia> y = [:H, :H, :S, :H]
4-element Vector{Symbol}:
 :H
 :H
 :S
 :H

julia> _group_by_class(df, y)
2×3 DataFrame
 Row │ firstcol   secondcol   class
     │ Array…     Array…      Symbol
─────┼───────────────────────────────
   1 │ [1, 2, 4]  [8, 9, 10]  H
   2 │ [3]        [7]         S
```
"""
function _group_by_class(df::AbstractDataFrame, y::AbstractVector{<:Union{String, Symbol}})
    ndf = DataFrame()
    classes = unique(y)
    attrsname = names(df)
    for attr in attrsname
        coltype = eltype(df[:, attr])
        col = Vector{Vector{coltype}}()
        for cls in classes
            idxes = findall(==(cls), y)
            push!(col, df[idxes, attr])
        end
        insertcols!(ndf, attr => col)
    end
    insertcols!(ndf, :class => classes) # insert class column
    return ndf
end

"""
# Examples
```julia-repl
julia> df = DataFrame(:firstcol => [1,2,3,4], :secondcol => [8,9,7,10], :myclasses => [:H, :H, :S, :H])
4×3 DataFrame
 Row │ firstcol  secondcol  myclasses
     │ Int64     Int64      Symbol
─────┼─────────────────────────────
   1 │        1          8  H
   2 │        2          9  H
   3 │        3          7  S
   4 │        4         10  H

julia> gdf = _group_by_class(df, "myclasses")
2×3 DataFrame
 Row │ firstcol   secondcol   class
     │ Array…     Array…      Symbol
─────┼───────────────────────────────
   1 │ [1, 2, 4]  [8, 9, 10]  H
   2 │ [3]        [7]         S
```
"""
function _group_by_class(df::AbstractDataFrame, class_colname::String)
    return _group_by_class(df[:, Not(class_colname)], df[:, class_colname])
end
