struct VarianceRanking <: AbstractFilterBased
    k::Int64

    function VarianceRanking(k::Int64)
        @assert k >= 0 "k must be greater or equal 0"
        new(k)
    end
end

selector_k(selector::VarianceRanking) = selector.k
selector_rankfunct(selector::VarianceRanking) = StatsBase.var

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceRanking;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceRanking,
    frame_index::Int;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector, frame_index; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::VarianceRanking,
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
    selector::VarianceRanking;
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
    selector::VarianceRanking,
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
    selector::VarianceRanking,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    for i in frame_indices
        apply!(mfd, selector, i, normalize_function=normalize_function)
    end
end

function build_bitmask(df::AbstractDataFrame, selector::VarianceRanking)::BitVector
    ranks = map(x->(x[1], selector_rankfunct(selector)(collect(Iterators.flatten(x[2])))), enumerate(eachcol(df)))

    # TODO: improve NaN management
    function lt(x, y)
        if isnan(x) && !isnan(y)
            return true
        elseif !isnan(x) && isnan(y)
            return false
        end
        return x < y
    end
    sort!(ranks, by=x->x[2], lt=lt)

    n_cols = ncol(df)
    if selector_k(selector) < n_cols
        bm = falses(n_cols)
        lower = n_cols - selector_k(selector) + 1
        for r in ranks[lower:n_cols]
            bm[r[1]] = true
        end
    else
        bm = trues(n_cols)
    end
    return bm
end
