# ---------------------------------------------------------------------------- #
#                              min/max normalize                               #
# ---------------------------------------------------------------------------- #
"""
Normalize passed DataFrame using min-max normalization.
Return a new normalized DataFrame
"""
minmax_normalize(c, args...; kwars...) = minmax_normalize!(deepcopy(c), args...; kwars...)

function minmax_normalize!(
    md::MultiData.MultiDataset,
    frame_index::Integer;
    min_quantile::Real = 0.0,
    max_quantile::Real = 1.0,
    col_quantile::Bool = true,
)
    return minmax_normalize!(
        MultiData.modality(md, frame_index);
        min_quantile = min_quantile,
        max_quantile = max_quantile,
        col_quantile = col_quantile
    )
end

function minmax_normalize!(
    X::AbstractMatrix;
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

    icols = eachcol(X)

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
    return X
end

function minmax_normalize!(
    df::AbstractDataFrame;
    kwargs...
)
    minmax_normalize!(Matrix(df); kwargs...)
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
    if (min == max)
        return repeat([0.5], length(v))
    end
    min = float(min)
    max = float(max)
    max = 1 / (max - min)
    rt = StatsBase.UnitRangeTransform(1, 1, true, [min], [max])
    # This function doesn't accept Integer
    return StatsBase.transform!(rt, v)
end

# ---------------------------------------------------------------------------- #
#                               normalize dataset                              #
# ---------------------------------------------------------------------------- #
"""
    _normalize_dataset!(
        X::AbstractMatrix{T},
        featid::Vector{<:SoleFeatures.InfoFeat};
        min_quantile::AbstractFloat=0.00,
        max_quantile::AbstractFloat=1.00,
        group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
    ) where {T<:Number}

Normalize the dataset matrix `X` by applying min-max normalization to groups of features.

## Parameters
- `X`: The input matrix to be normalized in-place
- `featid`: A vector of feature information objects that contain metadata about each feature
- `min_quantile`: The quantile to use as the minimum value (default: 0.00)
  - When set to 0.00, uses the absolute minimum value
  - Higher values (e.g., 0.05) ignore lower outliers by using the specified quantile instead
- `max_quantile`: The quantile to use as the maximum value (default: 1.00)
  - When set to 1.00, uses the absolute maximum value
  - Lower values (e.g., 0.95) ignore upper outliers by using the specified quantile instead
- `group`: A tuple of symbols representing fields in the `InfoFeat` objects to group by (default: (:nwin, :feat))
  - Features with the same values for these fields will be normalized together
  - For example, with the default (:nwin, :feat), features from the same window and of the same type
    will be normalized as a group, preserving their relative scale

## Details
The function performs group-wise normalization, which is essential when working with features that 
should maintain their relative scales. For example, when working with time series data, different 
measures (min, max, mean) applied to the same window should be normalized together to preserve 
their relationships.
"""
function _normalize_dataset(
    X::AbstractMatrix{T},
    featid::Vector{DataTreatments.FeatureId};
    min_quantile::AbstractFloat=0.00,
    max_quantile::AbstractFloat=1.00,
    group::Tuple{Vararg{Symbol}}=(:nwin, :feat),
) where {T<:Number}
    for g in _features_groupby(featid, group)
        minmax_normalize!(
            view(X, :, g);
            min_quantile = min_quantile,
            max_quantile = max_quantile,
            col_quantile = false
        )
    end
end

function _normalize_dataset(Xdf::AbstractDataFrame, featid::Vector{DataTreatments.FeatureId}; kwargs...)
    original_names = names(Xdf)
    DataFrame(_normalize_dataset!(Matrix(Xdf), featid; kwargs...), original_names)
end

function _features_groupby(
    featid::Vector{DataTreatments.FeatureId},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int}}
    res = Dict{Any, Vector{Int}}()
    for (i, g) in enumerate(featid)
        key = Tuple(getproperty(g, field) for field in aggrby)
        push!(get!(res, key, Int[]), i)
    end
    return collect(values(res))  # Return the grouped indices
end