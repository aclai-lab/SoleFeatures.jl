# ---------------------------------------------------------------------------- #
#                                filter struct                                 #
# ---------------------------------------------------------------------------- #
"""
    MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}

Rank features for classification using the Minimum Redundancy Maximum Relevance (MRMR) algorithm.

The MRMR algorithm finds an optimal set of features that is mutually and maximally dissimilar
and can effectively represent the response variable. It balances two objectives:
1. **Maximum Relevance**: Selects features highly correlated with the target variable
2. **Minimum Redundancy**: Avoids selecting features that are highly correlated with each other

# Algorithm

The MRMR score for a feature is computed as:
```
MRMR(f) = Relevance(f, y) / Redundancy(f, S)
```

Where:
- `Relevance(f, y)`: Measures how informative feature `f` is about target `y`
- `Redundancy(f, S)`: Measures similarity between feature `f` and already selected features `S`

# Fields
- `limiter::T`: A limiter that defines the selection criterion to be applied to scores

# Statistical Methods

## Relevance Measures

### F-Statistic (default)
One-way ANOVA F-test statistic that measures the ratio of between-group to within-group variance:
```
F = MS_between / MS_within
```
where:
- `MS_between`: Mean square between groups (variance between class means)
- `MS_within`: Mean square within groups (pooled variance within classes)

Higher F-values indicate stronger association between feature and target.

### Kolmogorov-Smirnov Test
Non-parametric test measuring the maximum distance between cumulative distribution functions
of feature values for different classes:
```
D = sup_x |F_in(x) - F_out(x)|
```
where:
- `F_in(x)`: CDF of feature values in a class
- `F_out(x)`: CDF of feature values outside the class

Returns mean KS statistic across all classes. Values range [0,1], higher means better separation.

### Random Forest Importance
Uses ensemble decision trees to measure feature importance via impurity reduction:
```
Importance(f) = Σ (impurity decrease when splitting on f)
```
Averaged across all trees in the forest (100 trees, 70% sample rate, max depth 5).

## Redundancy Measures

### Pearson Correlation (default)
Measures linear correlation between features:
```
r = Cov(X_i, X_j) / (σ_i * σ_j)
```
Values range [-1, 1]. Absolute values close to 1 indicate high redundancy.

## Denominator Functions

### Mean (default)
Average redundancy with selected features:
```
Redundancy(f) = mean(|cor(f, s)| for s in S)
```

### Max
Maximum redundancy with any selected feature:
```
Redundancy(f) = max(|cor(f, s)| for s in S)
```

# References

- Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information
  criteria of max-dependency, max-relevance, and min-redundancy. IEEE TPAMI, 27(8), 1226-1238.
- [MRMR Explained](https://medium.com/data-science/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b)
- [Python Implementation](https://github.com/smazzanti/mrmr)

# Examples

```julia
# Default configuration (F-statistic relevance, Pearson correlation redundancy)
filter = MrMrFilter(TopK(10))
ranked_features = score(filter, X, y)

# Using Kolmogorov-Smirnov test for relevance
ranked_features = score(filter, X, y; relevance=kolmogorov_smirnov)

# Using maximum redundancy instead of mean
ranked_features = score(filter, X, y; denominator=max)

# Using Random Forest importance
ranked_features = score(filter, X, y; relevance=random_forest)
```

See also: [`f_statistic`](@ref), [`kolmogorov_smirnov`](@ref), [`random_forest`](@ref), [`mrmr_classif`](@ref)
"""
struct MrMrFilter{T <: AbstractLimiter} <: AbstractMrMrFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractMrMrFilter) = true
is_unsupervised(::AbstractMrMrFilter) = false

# ---------------------------------------------------------------------------- #
#                                 f_statistic                                  #
# ---------------------------------------------------------------------------- #
function _f_statistic(x::AbstractVector{T}, y::AbstractVector)::T where {T<:Real}
    # One-way ANOVA F-statistic
    classes = unique(y)
    nclasses, n = length(classes), length(x)

    class_sqr    = Vector{T}(undef, nclasses)
    @inbounds for (i, c) in enumerate(classes)
        mask = y .== c
        class_sqr[i]    = sum(@view x[mask])^2 / sum(mask)
    end

    grand_sqr  = sum(x)^2 / n

    ss_between = sum(class_sqr[i] for i in 1:nclasses) - grand_sqr
    ss_within  = sum(x.^2) - grand_sqr - ss_between

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

"""
    f_statistic(X::AbstractArray, y::AbstractVector) -> Vector{Float64}

Compute one-way ANOVA F-statistic for each feature in X with respect to target y.

The F-statistic measures the ratio of between-group variance to within-group variance:

```
F = (SS_between / df_between) / (SS_within / df_within)
```

where:
- `SS_between`: Sum of squares between groups (explained variance)
- `SS_within`: Sum of squares within groups (unexplained variance)  
- `df_between = k - 1`: degrees of freedom (k = number of classes)
- `df_within = n - k`: degrees of freedom (n = number of samples)

# Arguments
- `X::AbstractArray`: Feature matrix (n_samples × n_features) or single vector
- `y::AbstractVector`: Target labels

# Returns
- `Vector{Float64}`: F-statistic for each feature (higher = more discriminative)

# Statistical Interpretation
- F = 0: No relationship between feature and target
- F > 0: Positive relationship (higher = stronger)
- Returns 0 if within-group variance is zero (perfect separation)

# Examples
```julia
# Single feature
f_score = f_statistic(X[:, 1], y)

# All features
f_scores = f_statistic(X, y)
```
"""
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

"""
    kolmogorov_smirnov(X::AbstractArray, y::AbstractVector) -> Vector{Float64}

Compute Kolmogorov-Smirnov test statistic for each feature in X with respect to target y.

The KS test measures the maximum distance between the cumulative distribution functions
of a feature's values inside and outside each class:

```
D = sup_x |F_in(x) - F_out(x)|
```

The final score is the mean KS statistic across all classes.

# Arguments
- `X::AbstractArray`: Feature matrix (n_samples × n_features)
- `y::AbstractVector`: Target labels

# Returns  
- `Vector{Float64}`: Mean KS statistic for each feature, range [0, 1]
  - 0: Identical distributions (no discrimination)
  - 1: Completely separated distributions (perfect discrimination)

# Statistical Interpretation
Non-parametric test that doesn't assume normal distributions. Particularly useful for:
- Ordinal or skewed features
- When class distributions have different shapes (not just different means)
- Detecting distributional differences beyond location shifts

# Examples
```julia
# Single feature
ks_score = kolmogorov_smirnov(X[:, 1], y)

# All features  
ks_scores = kolmogorov_smirnov(X, y)
```
"""
kolmogorov_smirnov()::Function = x, y -> _kolmogorov_smirnov(x, y)
kolmogorov_smirnov(X::AbstractArray, y::AbstractVector) = _kolmogorov_smirnov(X, y)

# ---------------------------------------------------------------------------- #
#                                 random_forest                                #
# ---------------------------------------------------------------------------- #
function _random_forest(X::AbstractArray{T}, y::AbstractVector;)::Vector{T} where {T<:Real}
    forest = DecisionTree.build_forest(y, X, -1, 100, 0.7, 5; rng=0)
    return DecisionTree.impurity_importance(forest)
end

"""
    random_forest(X::AbstractArray, y::AbstractVector) -> Vector{Float64}

Compute feature importance scores using Random Forest impurity reduction.

Trains a Random Forest classifier and measures each feature's importance as the
total reduction in node impurity (Gini impurity) when splitting on that feature,
averaged across all trees in the forest.

# Arguments
- `X::AbstractArray`: Feature matrix (n_samples × n_features)
- `y::AbstractVector`: Target labels

# Returns
- `Vector{Float64}`: Importance score for each feature (higher = more important)

# Forest Parameters
- Number of trees: 100
- Sample rate: 0.7 (70% of data per tree)
- Maximum depth: 5
- Features per split: All available (√n_features commonly used)
- Random seed: 0 (for reproducibility)

# Statistical Interpretation
Importance = Σ(weighted impurity decrease when splitting on feature)

- Captures non-linear relationships and interactions
- Less sensitive to feature scaling than correlation-based methods
- Can detect complex patterns that linear methods miss
- May favor features with more categories or higher cardinality

# Examples
```julia
# Compute RF importance
rf_scores = random_forest(X, y)
```
"""
random_forest()::Function = x, y -> _random_forest(x, y)
random_forest(X::AbstractArray, y::AbstractVector) = _random_forest(X, y)

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
        scores[rel_select] = rel_result[rel_select] ./ score_denominator[rel_select]
        active_idxs = findall(rel_select)
        idx_in_active = argmax(scores[rel_select])
        idx = active_idxs[idx_in_active]
        rel_select[idx] = 0
        push!(result, idx)
    end

    rel_result = relevance(X, y)
    nfeats = length(rel_result)
    scores = zeros(T, nfeats)
    rel_select = BitVector(ones(Bool, nfeats))
    score_denominator = ones(T, nfeats)
    result = Int64[]

    for _ in 1:nfeats
        select_feature!(scores, rel_result, rel_select, score_denominator)
        score_denominator[rel_select] = denominator(abs.(redundancy(X[:,rel_select], X[:,.!rel_select])), dims=2)
    end

    return result
end

"""
    mrmr_classif(X, y; relevance=f_statistic, redundancy=cor, denominator=mean)

Perform Minimum Redundancy Maximum Relevance (MRMR) feature selection for classification.

# Arguments
- `X::AbstractArray`: Feature matrix (n_samples × n_features)
- `y::AbstractVector`: Target labels (will be converted to integer codes if needed)

# Keyword Arguments
- `relevance::Function`: Function to compute feature-target association
  - Options: `f_statistic` (default), `kolmogorov_smirnov`, `random_forest`
- `redundancy::Function`: Function to compute feature-feature association  
  - Default: `StatsBase.cor` (Pearson correlation)
- `denominator::Function`: Aggregation function for redundancy scores
  - Options: `mean` (default), `max`

# Returns
- `Vector{Int}`: Feature indices ranked by MRMR score (most relevant first)

# Algorithm Steps
1. Compute relevance scores for all features
2. Select feature with highest relevance
3. For each remaining feature:
   - Compute redundancy with already selected features
   - Calculate MRMR score: relevance / aggregate_redundancy
   - Select feature with highest MRMR score
4. Repeat until all features ranked

# Examples
```julia
# Default: F-statistic relevance, mean correlation redundancy
ranked = mrmr_classif(X, y)

# Non-parametric relevance measure
ranked = mrmr_classif(X, y; relevance=kolmogorov_smirnov)

# Penalize maximum redundancy instead of mean
ranked = mrmr_classif(X, y; denominator=max)

# Complex relationships with Random Forest
ranked = mrmr_classif(X, y; relevance=random_forest)
```
"""
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