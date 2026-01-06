# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    KSTestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}

Univariate Kolmogorov–Smirnov (KS) filter for classification.  
For each feature column, it runs a (approximate) two-sample KS test between
the samples of each class vs. the rest, then averages the KS statistics.

Features are ranked by their KS statistic (higher means more separation across
classes).

# Arguments
- `X::AbstractArray{T}`: Matrix of shape (n_samples, n_features). Converted to
  floating point internally if not already `AbstractFloat`.
- `y::AbstractVector`: Class labels vector of shape `(n_samples,)`

# Fields
- `rank::Vector{Int64}`: Feature indices sorted by descending KS statistic.
- `score::Vector{<:Real}`: KS statistic per feature.

# Notes
- Uses `HypothesisTests.ApproximateTwoSampleKSTest` per feature/class split.
- Expects nonempty groups for each class to compute a statistic; otherwise the
  feature receives score 0.0 for that class.

# Example
```julia
X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
ksscore = KSTestFilter(X, y)
```
"""
struct KSTestFilter{F<:Real,T<:AbstractTask,L<:AbstractLearning,D<:AbstractDimensionality} <: AbstractFilter{F,T,L,D}
    rank  :: Vector{Int64}
    score :: Vector{F}

    function KSTestFilter(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
        y isa AbstractVector{<:Integer} || (y=CategoricalArrays.levelcode.(y))
        T isa AbstractFloat             || (X=float.(X))
        rank, score = _kolmogorov_smirnov(X, y)
        new{eltype(score),ClassificationTask,Supervised,Univariate}(rank, score)
    end
end

# ---------------------------------------------------------------------------- #
#                              kolmogorov_smirnov                              #
# ---------------------------------------------------------------------------- #
function _kolmogorov_smirnov(X::AbstractArray{T}, y::AbstractVector) where {T<:Real}
    nfeatures = size(X,2)
    ks_result = Vector{T}(undef, nfeatures)

    Threads.@threads for i in 1:nfeatures
        ks_result[i] = _ks_vec(X[:,i], y)
    end

    return sortperm(ks_result, rev=true), ks_result
end

function _ks_vec(x::AbstractVector{T}, y::AbstractVector) where {T<:Real}
    classes = unique(y)
    scores  = T[]
    
    for c in classes
        x_in_group  = x[y .== c]
        x_out_group = x[y .!= c]
        
        if length(x_in_group) > 0 && length(x_out_group) > 0
            ks_stat = HypothesisTests.ApproximateTwoSampleKSTest(x_in_group, x_out_group).δ
            push!(scores, ks_stat)
        end
    end
    
    return isempty(scores) ? zero(T) : mean(scores)
end
