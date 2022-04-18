using StatsBase

struct VarianceThreshold <: AbstractUnivariateSelector
    threshold::Float64

    function VarianceThreshold(threshold::Float64)
        @assert 0.0 <= threshold <= 1.0 "Threshold must be within [0,1]"
        new(threshold)
    end
end

selector_threshold(selector::VarianceThreshold) = selector.threshold
selector_function(selector::VarianceThreshold) = StatsBase.var

"""
Return a new MultiFrameDataset without the attributes considered unsitable using variance threshold.

## EXAMPLE
```jldoctest
julia> mfd = SoleBase.MultiFrameDataset(DataFrame(:firstCol => [[0,5.2],[2,13.87],[-3,7]],
                                            :secondCol => [[1.5,2],[2.2,1.2,1],[1,3,1.7]]))
● MultiFrameDataset
   └─ dimensions: ()
- Spare attributes
   └─ dimension: 1
3×2 SubDataFrame
 Row │ firstCol      secondCol
     │ Array…        Array…
─────┼───────────────────────────────
   1 │ [0.0, 5.2]    [1.5, 2.0]
   2 │ [2.0, 13.87]  [2.2, 1.2, 1.0]
   3 │ [-3.0, 7.0]   [1.0, 3.0, 1.7]


julia> vt = VarianceThreshold(0.12)
VarianceThreshold(0.12)

julia> SoleFeatures.apply(mfd, vt)
● MultiFrameDataset
   └─ dimensions: ()
- Spare attributes
   └─ dimension: 1
3×1 SubDataFrame
 Row │ firstCol
     │ Array…
─────┼──────────────
   1 │ [0.0, 5.2]
   2 │ [2.0, 13.87]
   3 │ [-3.0, 7.0]
```
"""
function apply(mfd::SoleBase.AbstractMultiFrameDataset, selector::VarianceThreshold)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector)
    return mfd_clone
end

function apply!(mfd::SoleBase.AbstractMultiFrameDataset, selector::VarianceThreshold)
    df = SoleBase.SoleDataset.data(mfd)
    @assert all(col->(col isa Array{<:Number}), collect(Iterators.flatten(eachcol(df))))
        "Attributes are not numerical type"
    df_norm = _minmax_normalize(df)
    bm = build_bit_mask(df_norm, selector)
    indicies = findall(x->!x, bm)
    SoleBase.SoleDataset.dropattributes!(mfd, indicies)
end

function build_bit_mask(
    df::DataFrame,
    selector::VarianceThreshold
)::BitVector
    return map(x->(selector_function(selector)(collect(Iterators.flatten(x))) >= selector_threshold(selector)), eachcol(df))
end

"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
function _minmax_normalize(df::DataFrame)::DataFrame
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
