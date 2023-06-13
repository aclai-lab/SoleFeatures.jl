# TODO: minmax_normalize in SoleData
"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
minmax_normalize(c, args...; kwars...) = minmax_normalize!(deepcopy(c), args...; kwars...)

function minmax_normalize!(
    mfd::SoleData.MultiModalDataset,
    frame_index::Integer;
    min_quantile::Real = 0.0,
    max_quantile::Real = 1.0,
    col_quantile::Bool = true,
)
    return minmax_normalize!(
        SoleData.frame(mfd, frame_index);
        min_quantile = min_quantile,
        max_quantile = max_quantile,
        col_quantile = col_quantile
    )
end

function minmax_normalize!(
    df::AbstractDataFrame;
    min_quantile::Real = 0.0,
    max_quantile::Real = 1.0,
    col_quantile::Bool = true,
)
    min_quantile < 0.0 &&
        throw(DomainError(min_quantile, "min_quantile must be greater than or equal to 0"))
    max_quantile > 1.0 &&
        throw(DomainError(max_quantile, "max_quantile must be less than or equal to 1"))
    max_quantile <= min_quantile &&
        throw(DomainError("max_quantile must be greater then min_quantile"))

    icols = eachcol(df)

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
    v::AbstractArray{<:AbstractArray{<:Real}},
    min::Real,
    max::Real
)
    return minmax_normalize!.(v, min, max)
end

function minmax_normalize!(
    v::AbstractArray{<:Real},
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
    _fr_bm2mfd_bm(mfd, frame_index, frame_bm)

frame bitmask to MultiModalDataset bitmask.

return bitmask for entire MultiModalDataset from a frame of it
"""
function _fr_bm2mfd_bm(
    mfd::SoleData.MultiModalDataset,
    frameidxes::Union{Integer, AbstractVector{<:Integer}},
    framebms::Union{BitVector, AbstractVector{<:BitVector}}
)::BitVector
    frameidxes = [ frameidxes... ]
    isa(framebms, BitVector) && (framebms = [ framebms ])

    length(frameidxes) != length(framebms) && throw(DimensionMismatch(""))

    bm = trues(nvariables(mfd))
    for i in 1:lastindex(frameidxes)
        fridx = frameidxes[i]
        frbm = framebms[i]
        framedescr = SoleData.grouped_variables(mfd)[fridx] # frame indices inside mfd
        bm[framedescr] = frbm
    end
    return bm
end

"""
    bm2attr

return tuple containing names of suitable attributes and names of not suitable attributes
"""
function bm2attr(mfd::SoleData.MultiModalDataset, bm::BitVector)
    return bm2attr(SoleData.data(mfd), bm)
end

function bm2attr(mfd::SoleData.MultiModalDataset, fridx::Integer, bm::BitVector)
    return bm2attr(SoleData.frame(mfd, fridx), bm)
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
function _group_by_class(df::AbstractDataFrame, y::AbstractVector{<:Class})
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
