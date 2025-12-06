using SoleFeatures

using MLJ
using RDatasets
iris = dataset("datasets", "iris")
X = Matrix(iris[:,1:end-1])
y = iris[:,end]
y = MLJ.levelcode.(y)

mrmr_classif(X, y; relevance=random_forest)

#####################################################################################
using DataFrames
using Statistics
using Distributed
using HypothesisTests
using DecisionTree
using StatsBase

# Parallel processing helper
function parallel_df(func, df::DataFrame, series::Vector, n_jobs::Int)
    n_jobs = n_jobs == -1 ? min(Sys.CPU_THREADS, ncol(df)) : min(Sys.CPU_THREADS, n_jobs)
    col_chunks = collect(Iterators.partition(1:ncol(df), ceil(Int, ncol(df) / n_jobs)))
    
    results = pmap(col_chunk -> func(df[:, col_chunk], series), col_chunks)
    return vcat(results...)
end

# F-statistic for classification
function _f_classif_series(x::Vector, y::Vector)
    not_na = .!ismissing.(x)
    if sum(not_na) == 0
        return 0.0
    end
    x_clean = x[not_na]
    y_clean = y[not_na]
    
    # One-way ANOVA F-statistic
    groups = unique(y_clean)
    k = length(groups)
    n = length(x_clean)
    
    grand_mean = mean(x_clean)
    ss_between = sum([sum(y_clean .== g) * (mean(x_clean[y_clean .== g]) - grand_mean)^2 for g in groups])
    ss_within = sum([(xi - mean(x_clean[y_clean .== yi]))^2 for (xi, yi) in zip(x_clean, y_clean)])
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    # return ms_within == 0 ? 0.0 : ms_between / ms_within
    prob = [fdtrc(dfbn, dfwn, f_val) for f_val in f]
    
    return f, prob
end

# Helper function for F-distribution survival function (complementary CDF)
# Equivalent to scipy.special.fdtrc
function fdtrc(dfn::Real, dfd::Real, x::Real)
    """
    F distribution survival function (1 - CDF).
    
    This is equivalent to scipy.special.fdtrc.
    Uses the regularized incomplete beta function.
    """
    if x <= 0
        return 1.0
    end
    
    # Using the relationship between F-distribution and beta distribution
    # P(F > x) = I_{dfd/(dfd + dfn*x)}(dfd/2, dfn/2)
    # where I is the regularized incomplete beta function
    
    w = dfd / (dfd + dfn * x)
    return beta_inc(dfd / 2, dfn / 2, w)[1]
end


function f_classif(X::DataFrame, y::Vector; n_jobs::Int=-1)
    return [_f_classif_series(X[:, i], y) for i in 1:ncol(X)]
end

# F-statistic for regression
function _f_regression_series(x::Vector, y::Vector)
    not_na = .!ismissing.(x) .& .!ismissing.(y)
    if sum(not_na) == 0
        return 0.0
    end
    x_clean = x[not_na]
    y_clean = y[not_na]
    
    # Pearson correlation squared * (n-2) / (1 - r^2)
    r = cor(x_clean, y_clean)
    n = length(x_clean)
    r_squared = r^2
    
    return (1 - r_squared) == 0 ? 0.0 : (r_squared * (n - 2)) / (1 - r_squared)
end

function f_regression(X::DataFrame, y::Vector; n_jobs::Int=-1)
    return [_f_regression_series(X[:, i], y) for i in 1:ncol(X)]
end

# Kolmogorov-Smirnov for classification
function _ks_classif_series(x::Vector, y::Vector)
    not_na = .!ismissing.(x)
    if sum(not_na) == 0
        return 0.0
    end
    x_clean = x[not_na]
    y_clean = y[not_na]
    
    groups = unique(y_clean)
    scores = Float64[]
    
    for g in groups
        x_in_group = x_clean[y_clean .== g]
        x_out_group = x_clean[y_clean .!= g]
        
        if length(x_in_group) > 0 && length(x_out_group) > 0
            ks_stat = ApproximateTwoSampleKSTest(x_in_group, x_out_group).δ
            push!(scores, ks_stat)
        end
    end
    
    return isempty(scores) ? 0.0 : mean(scores)
end

function ks_classif(X::DataFrame, y::Vector; n_jobs::Int=-1)
    return [_ks_classif_series(X[:, i], y) for i in 1:ncol(X)]
end

# Random Forest feature importance
function random_forest_classif(X::DataFrame, y::Vector)
    X_matrix = Matrix{Float64}(coalesce.(X, minimum(skipmissing(Matrix(X))) - 1))
    model = DecisionTreeClassifier(max_depth=5)
    fit!(model, X_matrix, y)
    
    # Feature importance not directly available in DecisionTree.jl
    # Return uniform importance as placeholder
    return ones(ncol(X)) ./ ncol(X)
end

function random_forest_regression(X::DataFrame, y::Vector)
    X_matrix = Matrix{Float64}(coalesce.(X, minimum(skipmissing(Matrix(X))) - 1))
    model = DecisionTreeRegressor(max_depth=5)
    fit!(model, X_matrix, y)
    
    return ones(ncol(X)) ./ ncol(X)
end

# Correlation
function correlation(target_column::String, features::Vector{String}, X::DataFrame; n_jobs::Int=-1)
    target = X[:, target_column]
    return [cor(skipmissing(target), skipmissing(X[:, f])) for f in features]
end

# Main MRMR functions
function mrmr_classif(
    X::DataFrame, y::Vector, K::Int;
    relevance::Union{String, Function}="f",
    redundancy::Union{String, Function}="c",
    denominator::Union{String, Function}="mean",
    cat_features::Union{Nothing, Vector{String}}=nothing,
    only_same_domain::Bool=false,
    return_scores::Bool=false,
    n_jobs::Int=-1,
    show_progress::Bool=true
)
    # Encoding categorical features (simplified - implement proper encoding if needed)
    # if cat_features !== nothing
    #     # Implement encoding here
    # end
    
    relevance_func = if relevance == "f"
        (X, y) -> f_classif(X, y; n_jobs=n_jobs)
    elseif relevance == "ks"
        (X, y) -> ks_classif(X, y; n_jobs=n_jobs)
    elseif relevance == "rf"
        random_forest_classif
    else
        relevance
    end
    
    redundancy_func = redundancy == "c" ? 
        (target, features, X) -> correlation(target, features, X; n_jobs=n_jobs) : 
        redundancy
    
    denominator_func = denominator == "mean" ? mean : 
                      (denominator == "max" ? maximum : denominator)
    
    relevance_args = (X=X, y=y)
    redundancy_args = (X=X,)
    
    return mrmr_base(
        K=K,
        relevance_func=relevance_func,
        redundancy_func=redundancy_func,
        relevance_args=relevance_args,
        redundancy_args=redundancy_args,
        denominator_func=denominator_func,
        only_same_domain=only_same_domain,
        return_scores=return_scores,
        show_progress=show_progress
    )
end

function mrmr_regression(
    X::DataFrame, y::Vector, K::Int;
    relevance::Union{String, Function}="f",
    redundancy::Union{String, Function}="c",
    denominator::Union{String, Function}="mean",
    cat_features::Union{Nothing, Vector{String}}=nothing,
    only_same_domain::Bool=false,
    return_scores::Bool=false,
    n_jobs::Int=-1,
    show_progress::Bool=true
)
    relevance_func = relevance == "f" ? 
        (X, y) -> f_regression(X, y; n_jobs=n_jobs) :
        (relevance == "rf" ? random_forest_regression : relevance)
    
    redundancy_func = redundancy == "c" ?
        (target, features, X) -> correlation(target, features, X; n_jobs=n_jobs) :
        redundancy
    
    denominator_func = denominator == "mean" ? mean :
                      (denominator == "max" ? maximum : denominator)
    
    relevance_args = (X=X, y=y)
    redundancy_args = (X=X,)
    
    return mrmr_base(
        K=K,
        relevance_func=relevance_func,
        redundancy_func=redundancy_func,
        relevance_args=relevance_args,
        redundancy_args=redundancy_args,
        denominator_func=denominator_func,
        only_same_domain=only_same_domain,
        return_scores=return_scores,
        show_progress=show_progress
    )
end

using DataFrames
using Statistics
using ProgressMeter

const FLOOR = 0.001

function groupstats2fstat(avg::DataFrame, var::DataFrame, n::DataFrame)
    """
    Compute F-statistic of some variables across groups

    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.

    Parameters
    ----------
    avg: DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.

    var: DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.

    n: DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.

    Returns
    -------
    f: Vector of shape (n_variables, )
        F-statistic of each variable, based on group statistics.

    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = sum(Matrix(avg) .* Matrix(n), dims=1) ./ sum(Matrix(n), dims=1)  # global average of each variable
    numerator = sum(Matrix(n) .* ((Matrix(avg) .- avg_global) .^ 2), dims=1) ./ (nrow(n) - 1)  # between group variability
    denominator = sum(Matrix(var) .* Matrix(n), dims=1) ./ (sum(Matrix(n), dims=1) .- nrow(n))  # within group variability
    f = vec(numerator ./ denominator)
    return coalesce.(f, 0.0)
end

function mrmr_base(;
    K::Int,
    relevance_func::Function,
    redundancy_func::Function,
    relevance_args::NamedTuple=NamedTuple(),
    redundancy_args::NamedTuple=NamedTuple(),
    denominator_func::Function=mean,
    only_same_domain::Bool=false,
    return_scores::Bool=false,
    show_progress::Bool=true
)
    # Compute relevance
    relevance_result = relevance_func(; relevance_args...)
    
    # Convert to Dict if needed
    if relevance_result isa AbstractVector
        relevance = Dict(string(i) => v for (i, v) in enumerate(relevance_result))
    else
        relevance = relevance_result
    end
    
    # Filter features with positive relevance
    features = [k for (k, v) in relevance if !ismissing(v) && v > 0]
    relevance = Dict(k => relevance[k] for k in features)
    
    # Initialize redundancy matrix
    redundancy = Dict((f1, f2) => FLOOR for f1 in features for f2 in features)
    
    K = min(K, length(features))
    selected_features = String[]
    not_selected_features = copy(features)

    prog = Progress(K, enabled=show_progress, desc="mRMR feature selection: ")
    
    for i in 1:K
        
        # Compute score numerator (relevance)
        score_numerator = Dict(f => relevance[f] for f in not_selected_features)

        if i > 1
            last_selected_feature = selected_features[end]

            # Determine which features to compare based on domain
            if only_same_domain
                last_domain = split(last_selected_feature, '_')[1]
                not_selected_features_sub = [c for c in not_selected_features 
                                            if split(c, '_')[1] == last_domain]
            else
                not_selected_features_sub = not_selected_features
            end

            # Compute redundancy for unselected features with last selected feature
            if !isempty(not_selected_features_sub)
                redundancy_result = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub;
                    redundancy_args...
                )
                
                # Update redundancy matrix
                for (j, f) in enumerate(not_selected_features_sub)
                    red_val = redundancy_result[j]
                    red_val = ismissing(red_val) ? FLOOR : abs(red_val)
                    red_val = max(red_val, FLOOR)
                    redundancy[(f, last_selected_feature)] = red_val
                end
            end

            # Compute score denominator (average redundancy with selected features)
            score_denominator = Dict{String, Float64}()
            for f in not_selected_features
                redundancies = [redundancy[(f, sf)] for sf in selected_features]
                denom_val = denominator_func(redundancies)
                score_denominator[f] = denom_val == 1.0 ? Inf : denom_val
            end
        else
            score_denominator = Dict(f => 1.0 for f in features)
        end

        # Compute mRMR score
        score = Dict(f => score_numerator[f] / score_denominator[f] 
                    for f in not_selected_features)

        # Select best feature
        best_feature = argmax(score)
        push!(selected_features, best_feature)
        filter!(x -> x != best_feature, not_selected_features)
        
        next!(prog)
    end

    if !return_scores
        return selected_features
    else
        return (selected_features, relevance, redundancy)
    end
end

#######################################################################
using Statistics
using LinearAlgebra
using SpecialFunctions

function f_oneway(x, y)    
    n_classes = unique(y)
    
    # Convert all inputs to Float64 arrays
    args = [Float64.(a) for a in args]
    
    # Number of samples per class
    n_samples_per_class = [size(a, 1) for a in args]
    n_samples = sum(n_samples_per_class)
    
    # Sum of squared values across all data
    ss_alldata = sum(sum(a .^ 2, dims=1) for a in args)
    
    # Sum of values for each group
    sums_args = [vec(sum(a, dims=1)) for a in args]
    
    # Square of sum across all data
    square_of_sums_alldata = sum(sums_args) .^ 2
    
    # Square of sums for each group
    square_of_sums_args = [s .^ 2 for s in sums_args]
    
    # Total sum of squares
    sstot = ss_alldata .- square_of_sums_alldata ./ Float64(n_samples)
    
    # Between-group sum of squares
    ssbn = 0.0
    for k in 1:length(args)
        ssbn += square_of_sums_args[k] ./ n_samples_per_class[k]
    end
    ssbn -= square_of_sums_alldata ./ Float64(n_samples)
    
    # Within-group sum of squares
    sswn = sstot .- ssbn
    
    # Degrees of freedom
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    
    # Mean squares
    msb = ssbn ./ Float64(dfbn)
    msw = sswn ./ Float64(dfwn)
    
    # Check for constant features
    constant_features_idx = findall(msw .== 0.0)
    if !isempty(constant_features_idx) && any(msb .!= 0.0)
        @warn "Features $constant_features_idx are constant."
    end
    
    # F-statistic
    f = msb ./ msw
    
    # Flatten to vector if needed
    f = vec(f)
    
    # Compute p-values using F-distribution survival function
    # In Julia, use the complementary CDF (ccdf) from Distributions.jl
    # or use SpecialFunctions.beta_inc for fdtrc equivalent
    prob = [fdtrc(dfbn, dfwn, f_val) for f_val in f]
    
    return f, prob
end

# Helper function for F-distribution survival function (complementary CDF)
# Equivalent to scipy.special.fdtrc
function fdtrc(dfn::Real, dfd::Real, x::Real)
    """
    F distribution survival function (1 - CDF).
    
    This is equivalent to scipy.special.fdtrc.
    Uses the regularized incomplete beta function.
    """
    if x <= 0
        return 1.0
    end
    
    # Using the relationship between F-distribution and beta distribution
    # P(F > x) = I_{dfd/(dfd + dfn*x)}(dfd/2, dfn/2)
    # where I is the regularized incomplete beta function
    
    w = dfd / (dfd + dfn * x)
    return beta_inc(dfd / 2, dfn / 2, w)[1]
end

###################################################################################à

function _f_statistic(x::AbstractVector{T}, y::AbstractVector) where {T<:Float64}
    # One-way ANOVA F-statistic
    groups = unique(y)
    k, n = length(groups), length(x)
    
    # Precompute grand mean
    grand_mean = mean(x)
    
    # Preallocate for group means and counts
    group_means = Vector{T}(undef, k)
    group_counts = Vector{Int}(undef, k)
    group_values = Vector{T}(undef, k)
    
    # Single pass to compute group statistics
    @inbounds for (i, g) in enumerate(groups)
        mask = y .== g
        group_counts[i] = sum(mask)
        group_means[i] = mean(@view x[mask])
        group_values[i] = sum(@view x[mask])
    end
    
    # Between-group sum of squares (vectorized)
    ss_between = sum(group_counts .* ((group_means .- grand_mean) .^ 2))
    
    # Within-group sum of squares (optimized single loop)
    ss_within = zero(T)
    @inbounds for (xi, yi) in zip(x, y)
        group_idx = findfirst(==(yi), groups)
        ss_within += (xi - group_means[group_idx])^2
    end
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

function _f_statistic(x::AbstractVector{T}, y::AbstractVector) where {T<:Float64}
    # One-way ANOVA F-statistic
    groups = unique(y)
    k, n = length(groups), length(x)
    
    grand_mean = sum(x.^2)
    grand_sqr = (sum(x)^2 - grand_mean) / n

    grp_sqrsum = Vector{T}(undef, k)
    @inbounds for (i, g) in enumerate(groups)
        mask = y .== g
        grp_sqrsum[i] = sum(@view x[mask])^2 / sum(mask)
    end
    ss_between = sum(grp_sqrsum) - grand_sqr
    ss_within = grand_mean - grand_sqr - ss_between
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    
    return iszero(ms_within) ? zero(T) : ms_between / ms_within
end

@btime collect(zip(x,y))

#################################################################################
nclasses = n_classes
x_sum = ss_alldata
class_counts = n_samples_per_class
n = n_samples
class_sum = sums_args
sqr_sum = square_of_sums_alldata
class_sqr = square_of_sums_args
ss_tot = ss_tot
ssbn = ssbn

function _f_statistic(x::AbstractVector{T}, y::AbstractVector) where {T<:Float64}
    # One-way ANOVA F-statistic
    classes = unique(y)
    nclasses, n = length(classes), length(x)
    x_sum = sum(x.^2)

    class_sum = Vector{T}(undef, nclasses)
    class_sqr = Vector{T}(undef, nclasses)
    class_counts = Vector{Int64}(undef, nclasses)
    ss_between = zero(T)
    @inbounds for (i, g) in enumerate(classes)
        mask = y .== g
        class_sum[i] = sum(@view x[mask])
        class_sqr[i] = class_sum[i]^2
        class_counts[i] = sum(mask)
        ss_between += class_sqr[i] / class_counts[i]
    end
    
    sqr_sum = sum(class_sum)^2
    ss_tot = x_sum - sqr_sum / n
    ss_between -= sqr_sum / n
    ss_within = ss_tot - ss_between

    dfbn = nclasses - 1
    dfwn = n - nclasses
    ms_between = ss_between / dfbn
    ms_within = ss_within / dfwn

    f = ms_between / ms_within
    
    # return iszero(ms_within) ? zero(T) : ms_between / ms_within
    prob = [fdtrc(dfbn, dfwn, f_val) for f_val in f]
    
    # return ms_between, ms_within, f, prob
    return x_sum, class_sum
end

# Helper function for F-distribution survival function (complementary CDF)
# Equivalent to scipy.special.fdtrc
function fdtrc(dfn::Real, dfd::Real, x::Real)
    if x <= 0
        return 1.0
    end
    
    w = dfd / (dfd + dfn * x)
    return beta_inc(dfd / 2, dfn / 2, w)[1]
end


# function _f_statistic(x::AbstractVector{T}, y::AbstractVector) where {T<:Float64}
#     # One-way ANOVA F-statistic
#     groups = unique(y)
#     k, n = length(groups), length(x)
    
#     stats = Dict(g => Mean() for g in groups)
#     @inbounds for (xi, yi) in zip(x, y)
#         fit!(stats[yi], xi)
#     end
    
#     grand_mean = mean(x)
    
#     ss_between = sum(nobs(stats[g]) * (value(stats[g]) - grand_mean)^2 for g in groups)
#     ss_within = sum((xi - value(stats[yi]))^2 for (xi, yi) in zip(x, y))
    
#     ms_between = ss_between / (k - 1)
#     ms_within = ss_within / (n - k)
    
#     return iszero(ms_within) ? zero(T) : ms_between / ms_within
# end

