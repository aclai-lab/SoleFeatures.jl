# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
# notes for Michy
# useful infos: https://medium.com/data-science/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
# reference: https://github.com/smazzanti/mrmr
"""
    MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}

Rank features for classification using minimum redundancy maximum relevance (MRMR) algorithm.
The MRMR algorithm finds an optimal set of features that is mutually and maximally dissimilar
and can represent the response variable effectively.
The algorithm minimizes the redundancy of a feature set and maximizes the relevance of a
feature set to the response variable.
The algorithm quantifies the redundancy and relevance using the mutual information of
variables—pairwise mutual information of features and mutual information of a feature and the response.

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores.

# Implementation Details
This algorithm is for classification problems.
"""
struct MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMrMrFilter) = true
is_unsupervised(::AbstractMrMrFilter) = false


function _correlation(X::AbstractMatrix{T}, target_idx::Int, feature_indices::Vector{Int}) where {T<:Real}
    target = @view X[:, target_idx]
    result = Dict{String,T}()
    
    for feat_idx in feature_indices
        feature = @view X[:, feat_idx]
        corr_val = Statistics.cor(feature, target)
        result[string(feat_idx)] = isnan(corr_val) ? zero(T) : corr_val
    end
    
    return result
end

function correlation(;
    target_column,
    features,
    X::AbstractMatrix{T}
) where {T<:Real}
    target_idx = parse(Int, target_column)
    feature_indices = [parse(Int, f) for f in features]
    
    return _correlation(X, target_idx, feature_indices)

end

# ---------------------------------------------------------------------------- #
#                                 f_statistic                                  #
# ---------------------------------------------------------------------------- #
function _f_statistic(x::AbstractVector{T}, y::AbstractVector)::T where {T<:Real}
    # One-way ANOVA F-statistic
    classes = unique(y)
    nclasses, n = length(classes), length(x)

    class_mean   = Vector{T}(undef, nclasses)
    class_counts = Vector{Int64}(undef, nclasses)
    @inbounds for (i, c) in enumerate(classes)
        mask = y .== c
        class_mean[i] = mean(@view x[mask])
        class_counts[i] = sum(mask)
    end
    
    grand_mean = mean(x)
    
    ss_between = sum(class_counts[i] * (class_mean[i] - grand_mean)^2 for (i, c) in enumerate(classes))
    ss_within = sum((xi - class_mean[yi])^2 for (xi, yi) in zip(x, y))
    
    ms_between = ss_between / (nclasses - 1)
    ms_within = ss_within / (n - nclasses)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

function _f_statistic(X::AbstractArray{T}, y::AbstractVector)::Vector{T}  where {T<:Real}
    nclasses = size(X,2)
    f_result = Vector{T}(undef, nclasses)
    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            f_result[i] = _f_statistic(X[:,i], y)
        end
    end
    return f_result
end

f_statistic()::Function = x, y -> _f_statistic(x, y)
f_statistic(X::AbstractArray, y::AbstractVector) = _f_statistic(X, y)

# ---------------------------------------------------------------------------- #
#                              kolmogorov_smirnov                              #
# ---------------------------------------------------------------------------- #
function _kolmogorov_smirnov(x::AbstractVector{T}, y::AbstractVector;)::T where {T<:Real}
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
    
    return isempty(scores) ? 0.0 : mean(scores)
end

function _kolmogorov_smirnov(X::AbstractArray{T}, y::AbstractVector)::Vector{T} where {T<:Real}
    nclasses = size(X,2)
    ks_result = Vector{T}(undef, nclasses)
    if nclasses > 10
        Threads.@threads for i in axes(X, 2)
            ks_result[i] = _kolmogorov_smirnov(X[:,i], y)
        end
    else
        for i in axes(X, 2)
            ks_result[i] = _kolmogorov_smirnov(X[:,i], y)
        end
    end
    return ks_result
end

kolmogorov_smirnov()::Function = x, y -> _kolmogorov_smirnov(x, y)
kolmogorov_smirnov(X::AbstractArray, y::AbstractVector) = _kolmogorov_smirnov(X, y)

# ---------------------------------------------------------------------------- #
#                                 random_forest                                #
# ---------------------------------------------------------------------------- #
function _random_forest(X::AbstractArray{T}, y::AbstractVector;)::Vector{T} where {T<:Real}
    forest = DecisionTree.build_forest(y, X, -1, 100, 0.7, 5; rng=0)
    return DecisionTree.impurity_importance(forest)
end

random_forest()::Function = x, y -> _random_forest(x, y)
random_forest(X::AbstractArray, y::AbstractVector) = _random_forest(X, y)
# correlation(;)::Function        = x -> _correlation(x; )

# ---------------------------------------------------------------------------- #
#              minimum redundancy maximum relevance classifier                 #
# ---------------------------------------------------------------------------- #
function _estimate_mrmr(
    X           :: AbstractArray{T}, 
    y           :: AbstractVector;
    relevance   :: Base.Callable,
    redundancy  :: Base.Callable,
    denominator :: Base.Callable
) where {T<:Real}
    function select_feature!(
        scores::Vector{T},
        rel_result::Vector{T},
        rel_select::BitVector,
        score_denominator::Vector{T},
    ) where T
        score = rel_result[rel_select] ./ score_denominator[rel_select]
        idx = argmax(rel_result)
        rel_result[idx] = zero(T)
        rel_select[idx] = 0
        scores[idx] = maximum(score)
    end

    rel_result = relevance(X, y)
    nfeats = length(rel_result)
    scores = zeros(T, nfeats)
    rel_select = BitVector(ones(Bool, nfeats))
    score_denominator = ones(T, nfeats)

    for _ in 1:nfeats
        select_feature!(scores, rel_result, rel_select, score_denominator)
        score_denominator[rel_select] = denominator(abs.(redundancy(X[:,rel_select], X[:,.!rel_select])), dims=2)
    end

    return scores
end

function mrmr_classif(
    X           :: AbstractArray{T}, 
    y           :: AbstractVector;
    relevance   :: Base.Callable=f_statistic,   # f_statistic, kolmogorov_smirnov, random_forest
    redundancy  :: Base.Callable=StatsBase.cor, # StatsBase.cor
    denominator :: Base.Callable=mean           # mean, max
# )::Vector{T} where {T<:Real}
) where {T<:Real}
    return _estimate_mrmr(X, y; relevance, redundancy, denominator)
end
mrmr_classif(X::AbstractArray{<:Real}, args...; kwargs...) = 
    mrmr_classif(Float64.(X), args...; kwargs...)

# ---------------------------------------------------------------------------- #
#                                    score                                     #
# ---------------------------------------------------------------------------- #
function score(
    ::MrMrFilter,
    X     :: AbstractArray{T},
    y     :: AbstractVector;
    kwargs...
)::Vector{T} where {T<:Real}
    y isa AbstractVector{<:Int} || (y=CategoricalArrays.levelcode.(y))
    return mrmr_classif(X, y; kwargs...)
end