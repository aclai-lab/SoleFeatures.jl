# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
# mimic the behavior of sklearn.preprocessing.StandardScaler
# function scale(X; dims=1, center=true, scale=true)
#     return StatsBase.standardize(ZScoreTransform, X, dims=dims, center=center, scale=scale)
# end

function scale(X; dims=1, center=true, scale=true)
    if scale && center
        return StatsBase.zscore(X, dims)
    elseif center && !scale
        # Center without scaling
        μ = mean(X, dims=dims)
        return X .- μ
    elseif !center && scale
        # Scale without centering
        σ = std(X, dims=dims)
        return X ./ σ
    else
        # Neither center nor scale
        return copy(X)
    end
end

# ---------------------------------------------------------------------------- #
#               mutual information discrete/discrete computation               #
# ---------------------------------------------------------------------------- #
"""
    compute_mi_dd(x, y)

Calculate mutual information between two discrete variables.
"""
function _compute_mi_dd(x, y)
    # Calculate joint probability distribution
    n_samples = length(x)
    x_values = unique(x)
    y_values = unique(y)
    
    joint_count = Dict{Tuple{eltype(x),eltype(y)},Int}()
    x_count = Dict{eltype(x),Int}()
    y_count = Dict{eltype(y),Int}()
    
    for i in 1:n_samples
        joint_count[(x[i], y[i])] = get(joint_count, (x[i], y[i]), 0) + 1
        x_count[x[i]] = get(x_count, x[i], 0) + 1
        y_count[y[i]] = get(y_count, y[i], 0) + 1
    end
    
    mi = 0.0
    for (xy, count) in joint_count
        px = x_count[xy[1]] / n_samples
        py = y_count[xy[2]] / n_samples
        pxy = count / n_samples
        mi += pxy * log(pxy / (px * py))
    end
    
    return max(0, mi)
end

# ---------------------------------------------------------------------------- #
#             mutual information continuous/continuous computation             #
# ---------------------------------------------------------------------------- #
"""
    compute_mi_cc(x, y, n_neighbors)

Compute mutual information between two continuous variables using
k-nearest neighbors method.

Based on Kraskov et al. (2004).
"""
function _compute_mi_cc(x, y, n_neighbors)
    n_samples = length(x)
    
    # Reshape into column vectors
    x_reshaped = reshape(x, :, 1)
    y_reshaped = reshape(y, :, 1)
    xy = hcat(x_reshaped, y_reshaped)
    
    # We'll need a kNN implementation - using NearestNeighbors.jl
    tree_xy = KDTree(xy', Chebyshev())
    idxs, dists = knn(tree_xy, xy', n_neighbors + 1)
    
    # Get radius as the distance to the k-th neighbor (excluding self)
    radius = [dists[i][end] for i in 1:n_samples]
    epsilon = nextfloat.(radius)
    
    # Count points within epsilon in x and y spaces
    tree_x = KDTree(x_reshaped', Chebyshev())
    tree_y = KDTree(y_reshaped', Chebyshev())
    
    nx = zeros(n_samples)
    ny = zeros(n_samples)
    
    for i in 1:n_samples
        nx[i] = length(inrange(tree_x, x_reshaped[i:i]', epsilon[i])) - 1
        ny[i] = length(inrange(tree_y, y_reshaped[i:i]', epsilon[i])) - 1
    end
    
    # Compute MI using the Kraskov formula
    mi = digamma(n_samples) + digamma(n_neighbors) - mean(digamma.(nx .+ 1)) - mean(digamma.(ny .+ 1))
    
    return max(0, mi)
end

# ---------------------------------------------------------------------------- #
#              mutual information continuous/discrete computation              #
# ---------------------------------------------------------------------------- #
"""
    compute_mi_cd(c, d, n_neighbors)

Compute mutual information between continuous and discrete variables.

Based on Ross (2014).
"""
function _compute_mi_cd(c, d, n_neighbors)
    n_samples = length(c)
    c_reshaped = reshape(c, :, 1)
    
    radius = zeros(n_samples)
    label_counts = zeros(n_samples)
    k_all = zeros(n_samples)
    
    # Process each unique discrete value
    for label in unique(d)
        mask = d .== label
        count = sum(mask)
        
        if count > 1
            k = min(n_neighbors, count - 1)
            
            # Get points with this label
            c_subset = c_reshaped[mask, :]
            
            # Build kd-tree and find distances
            tree = KDTree(c_subset', Chebyshev())
            idxs, dists = knn(tree, c_subset', k + 1)
            
            # Store radius as distance to k-th neighbor
            r = [dists[i][end] for i in 1:count]
            radius[mask] = nextfloat.(r)
            k_all[mask] .= k
        end
        
        label_counts[mask] .= count
    end
    
    # Ignore points with unique labels
    mask = label_counts .> 1
    n_valid_samples = sum(mask)
    
    # Skip calculation if we don't have enough valid samples
    if n_valid_samples == 0
        return 0.0
    end
    
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c_valid = c_reshaped[mask, :]
    radius_valid = radius[mask]
    
    # Count points within radius in the continuous space
    tree_c = KDTree(c_valid', Chebyshev())
    m_all = zeros(n_valid_samples)
    
    for i in 1:n_valid_samples
        m_all[i] = length(inrange(tree_c, c_valid[i:i]', radius_valid[i]))
    end
    
    # Compute MI using Ross formula
    mi = digamma(n_valid_samples) + mean(digamma.(k_all)) - 
         mean(digamma.(label_counts)) - mean(digamma.(m_all))
    
    return max(0, mi)
end

# ---------------------------------------------------------------------------- #
#                         mutual information estimator                         #
# ---------------------------------------------------------------------------- #
function _compute_mi(x, y, x_discrete, y_discrete, n_neighbors=3)
    if x_discrete && y_discrete
        return _compute_mi_dd(x, y)
    elseif x_discrete && !y_discrete
        return _compute_mi_cd(y, x, n_neighbors)
    elseif !x_discrete && y_discrete
        return _compute_mi_cd(x, y, n_neighbors)
    else
        return _compute_mi_cc(x, y, n_neighbors)
    end
end

"""Estimate mutual information between the features and the target.

Parameters
----------
X : AbstractDataFrame of shape (n_samples, n_features)
    Feature matrix.

y : array-like of shape (n_samples,)
    Target vector.

discrete_features : {:sparse, :auto, array-like}, default=:sparse
    If :auto, then determines whether to consider all features discrete
    or continuous. 
    If array, then it should be either a boolean mask
    with shape (n_features,) or array with indices of discrete features.
    If 'auto', it is assigned to False for dense `X` and to True for
    sparse `X`.

discrete_target : bool, default=False
    Whether to consider `y` as a discrete variable.

n_neighbors : int, default=3
    Number of neighbors to use for MI estimation for continuous variables,
    see [1]_ and [2]_. Higher values reduce variance of the estimation, but
    could introduce a bias.

copy : bool, default=True
    Whether to make a copy of the given data. If set to False, the initial
    data will be overwritten.

random_state : int, RandomState instance or None, default=None
    Determines random number generation for adding small noise to
    continuous variables in order to remove repeated values.
    Pass an int for reproducible results across multiple function calls.
    See :term:`Glossary <random_state>`.

n_jobs : int, default=None
    The number of jobs to use for computing the mutual information.
    The parallelization is done on the columns of `X`.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 1.5


Returns
-------
mi : ndarray, shape (n_features,)
    Estimated mutual information between each feature and the target in
    nat units. A negative value will be replaced by 0.

References
----------
.. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
       information". Phys. Rev. E 69, 2004.
.. [2] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
"""
function _estimate_mi(
    X::AbstractMatrix,
    y::AbstractVector{<:Class};
    discrete_mode::Union{AbstractArray, Nothing}=nothing,
    discrete_target::Bool=false,
    n_neighbors::Int=3,
    n_jobs::Union{Int, Nothing}=nothing,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    n_samples, n_features = size(X)

    # Handle discrete_features parameter
    if isnothing(discrete_mode)
        discrete_features = issparse(X)
        discrete_mask = fill(discrete_features, n_features)
    else
        discrete_mask = falses(n_features)
        discrete_mask[discrete_features] .= true
    end

    continuous_mask = .!discrete_mask
    # Check if sparse matrix has continuous features
    if any(continuous_mask) && issparse(X)
        throwArgumentError("Sparse matrix `X` can't have continuous features.")
    end

    # Handle continuous features
    any(continuous_mask) && begin
        # Scale continuous features
        X[:, continuous_mask] = scale(X[:, continuous_mask], dims=1, center=false)
        # Add small noise to continuous features
        means = max.(1, mean(abs.(X[:, continuous_mask]), dims=1))
        X[:, continuous_mask] .+= 1e-10 .* means .* randn(rng, n_samples, sum(continuous_mask))
    end
    
    # Handle continuous target
    discrete_target || begin
        y = scale(y, center=false)
        y .+= 1e-10 * max(1, mean(abs.(y))) * randn(rng, n_samples)
    end

    # Compute mutual information for each feature
    mi = Vector{Float64}(undef, n_features)
    
    # Use threading for parallel computation
    Threads.@threads for i in 1:n_features
        x = X[:, i]
        mi[i] = _compute_mi(x, y, discrete_mask[i], discrete_target, n_neighbors)
    end

    return mi
end

"""
    mutual_info_classif(
        X, 
        y; 
        discrete_features="auto", 
        n_neighbors=3, 
        copy=true, 
        random_state=nothing,
        n_jobs=nothing
    )

Estimate mutual information for a discrete target variable.

Mutual information (MI) between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal
to zero if and only if two random variables are independent, and higher
values mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances as described in Kraskov et al. (2004)
and Ross (2014).

# Parameters
- `X`: Feature matrix, shape (n_samples, n_features)
- `y`: Target vector, shape (n_samples,)
- `discrete_features`: If bool, determines whether to consider all features discrete
    or continuous. If array, should be either a boolean mask with shape (n_features,)
    or array with indices of discrete features. If "auto", is assigned to false for
    dense X and true for sparse X.
- `n_neighbors`: Number of neighbors to use for MI estimation for continuous variables.
- `copy`: Whether to make a copy of the given data. If false, initial data may be overwritten.
- `random_state`: Random seed or RNG for adding small noise to continuous variables.
- `n_jobs`: Number of jobs for parallel computation.

# Returns
- `mi`: Vector of estimated mutual information between each feature and the target in nat units.

# References
- A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual information". 
  Phys. Rev. E 69, 2004.
- B. C. Ross "Mutual Information between Discrete and Continuous Data Sets". 
  PLoS ONE 9(2), 2014.
"""
function mutual_info_classif(
    X::AbstractMatrix, 
    y::AbstractVector{<:Class}; 
    discrete_mode::Union{AbstractArray, Nothing}=nothing,
    n_neighbors::Int=3, 
    n_jobs=nothing,
    rng::AbstractRNG=Random.GLOBAL_RNG,
)    
    # Call _estimate_mi with discrete_target=true
    return _estimate_mi(
        X,
        y;
        discrete_mode,
        discrete_target=true,
        n_neighbors,
        n_jobs,
        rng
    )
end
