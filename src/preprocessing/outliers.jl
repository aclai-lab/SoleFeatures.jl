"""
An outlier is an observation that is very far from the mean of data set. Deleting these
values is very important because they can introduce errors or lead to incorrect reasoning.

In the state of the art, there are many methods to identify outliers, such as statistical
techniques, distance-based approaches, and machine learning algorithms.

In the following, we want to introduce some of these methods, in particular:
 - Z-Score
 - IQR Method
 - Isolation Forest
"""

struct OutlierResult{S<:AbstractVector}
    scores::S
    indices::Vector{Int}
end

outlier_scores(outlier_result::OutlierResult) = outlier_result.scores
outlier_indices(outlier_result::OutlierResult) = outlier_result.indices

"""
Z-Score Method
"""
function zscore(values::AbstractVector{T}) where {T<:Union{Missing,<:Real}}
    vals = collect(skipmissing(values))

    # If all values are missing, we return a vector with all missing
    isempty(vals) && return fill(missing, length(values))

    # Calculating mean and standard deviation of the distribution
    μ = mean(vals)
    σ = std(vals)

    return [(ismissing(v) ? missing : (v - μ) / σ) for v in values]
end

function zscore_outliers(
    values::AbstractVector{T};
    threshold::Float64 = 2.0
) where {T<:Union{Missing,<:Real}}
    scores = zscore(values)
    idxs = findall(x -> !ismissing(x) && abs(x) > threshold, scores)

    return OutlierResult(scores, idxs)
end

"""
IQR Method
"""
function iqr_outliers(
    values::AbstractVector{T};
    k::Float64 = 1.5
) where {T<:Union{Missing,<:Real}}
    vals = collect(skipmissing(values))

    isempty(vals) && return OutlierResult(fill(missing, length(values)), Int[])

    q1 = quantile(vals, 0.25)
    q3 = quantile(vals, 0.75)
    iqr_val = iqr(vals)

    lower = q1 - k * iqr_val
    upper = q3 + k * iqr_val

    idxs = findall(v -> !ismissing(v) && (v < lower || v > upper), values)

    return OutlierResult(fill(missing, length(values)), idxs)
end
