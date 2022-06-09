struct VarianceThreshold <: AbstractFilterBased
    threshold::Float64

    function VarianceThreshold(threshold::Float64)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
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
function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold,
    frame_index::Int;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector, frame_index; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    for i in frame_indices
        apply!(mfd_clone, selector, i, normalize_function=normalize_function)
    end
    return mfd_clone
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold;
    normalize_function=nothing
)
    df = SoleBase.SoleDataset.data(mfd)
    @assert all(col->(col isa Union{Array{<:Number},Number}),
                collect(Iterators.flatten(eachcol(df))))
                    "Attributes are not numerical type"

    if !isnothing(normalize_function)
        df_norm = normalize_function(df)
        bm = build_bitmask(df_norm, selector)
    else
        bm = build_bitmask(df, selector)
    end

    indices = findall(x->!x, bm)
    SoleBase.SoleDataset.dropattributes!(mfd, indices)
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold,
    frame_index::Int;
    normalize_function=nothing
)
    # frame from 'frame_index'
    fr = SoleBase.frame(mfd, frame_index)
    @assert all(col->(col isa Union{Array{<:Number},Number}),
                collect(Iterators.flatten(eachcol(fr))))
                    "Attributes are not numerical type"

    # frame indices
    fr_indices = SoleBase.SoleDataset.frame_descriptor(mfd)[frame_index]

    # check if the frame needs normalization
    if !isnothing(normalize_function)
        fr_norm = normalize_function(fr)
        fr_bm = build_bitmask(fr_norm, selector)
    else
        fr_bm = build_bitmask(fr, selector)
    end

    # bit mask for entire dataset
    bm = trues(nattributes(mfd))
    for i in 1:nattributes(fr)
        bm[fr_indices[i]] = fr_bm[i]
    end

    indices = findall(x->!x, bm)
    SoleBase.SoleDataset.dropattributes!(mfd, indices)
end

function apply!(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceThreshold,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    for i in frame_indices
        apply!(mfd, selector, i, normalize_function=normalize_function)
    end
end

function build_bitmask(
    df::AbstractDataFrame,
    selector::VarianceThreshold
)::BitVector
    return map(x->(selector_function(selector)(collect(Iterators.flatten(x))) >= selector_threshold(selector)), eachcol(df))
end
