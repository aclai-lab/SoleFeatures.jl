struct CorrelationThreshold <: AbstractFilterBased
    threshold::Float64
    cor_algorithm::Symbol

    function CorrelationThreshold(threshold::Float64, cor_algorithm::Symbol)
        if threshold < 0.0
            throw(DomainError(threshold, "threshold must be greater than or equal to 0"))
        end
        if !(cor_algorithm in [:pearson :spearman :kendall])
            throw(ErrorException("cor_algorithm must be :pearson, :spearman, :kendall"))
        end
        new(threshold, cor_algorithm)
    end
end

selector_threshold(selector::CorrelationThreshold) = selector.threshold
function selector_function(selector::CorrelationThreshold)
    if selector.cor_algorithm == :pearson
        return StatsBase.cor
    elseif selector.cor_algorithm == :spearman
        return StatsBase.corspearman
    elseif selector.cor_algorithm == :kendall
        return StatsBase.corkendall
    end
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::CorrelationThreshold;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::CorrelationThreshold,
    frame_index::Int;
    normalize_function=nothing
)
    mfd_clone = deepcopy(mfd)
    apply!(mfd_clone, selector, frame_index; normalize_function=normalize_function)
    return mfd_clone
end

function apply(
    mfd::SoleBase.AbstractMultiFrameDataset,
    selector::CorrelationThreshold,
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
    selector::CorrelationThreshold;
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
    selector::CorrelationThreshold,
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
    selector::CorrelationThreshold,
    frame_indices::AbstractVector{<:Int};
    normalize_function=nothing
)
    for i in frame_indices
        apply!(mfd, selector, i, normalize_function=normalize_function)
    end
end

function build_bitmask(
    df::AbstractDataFrame,
    selector::CorrelationThreshold
)::BitVector
    avg_vector = dtw_correlation(df, selector_function(selector))
    return BitVector(map(x->(x <= selector_threshold(selector)), avg_vector))
end
