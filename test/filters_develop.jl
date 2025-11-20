using MLJ
using StatsBase
using CategoricalArrays
using HypothesisTests
using MLBase
using Distributions
using SparseArrays
using Random
using NearestNeighbors

X, y = @load_iris
X = MLJ.matrix(X) 
y = MLJ.levelcode.(y)

function mutual_info_classifier(
    X           :: AbstractArray{T}, 
    y           :: AbstractVector; 
    n_neighbors :: Int64=3, 
    rng         :: AbstractRNG=Random.GLOBAL_RNG,
) where {T<:Float64}
    return _estimate_mi_classifier(X, y; n_neighbors, rng)
end
mutual_info_classifier(X::AbstractArray{<:Real}, args...; kwargs...) = 
    mutual_info_classifier(Float64.(X), args...; kwargs...)

function _estimate_mi_classifier(
    X           :: AbstractArray{T},
    y           :: AbstractVector;
    n_neighbors :: Int64,
    rng         :: AbstractRNG
) where {T<:Float64}
    scale!(X)
    # X .+= 1e-10 .* max.(1, mean(abs.(X), dims=1))
    X .+= 1e-10 .* max.(1, mean(abs.(X), dims=1)) .* randn(rng, size(X, 1))

    # compute mutual information for each feature
    mi = Vector{T}(undef, size(X, 2))

    # Threads.@threads for i in axes(X, 2)
    for i in axes(X, 2)
        mi[i] = _compute_mi_cd(X[:, i], y, n_neighbors)
    end

    return mi
end

# compute mutual information between continuous and discrete variables
function _compute_mi_cd(
    x::AbstractVector{T}, 
    y::AbstractVector{Int64},
    n_neighbors::Int64
) where {T<:Float64}
    n_samples    = length(x)
    radius       = Vector{T}(undef, n_samples)
    label_counts = Vector{T}(undef, n_samples)
    k_all        = Vector{T}(undef, n_samples)
    
    # Process each unique discrete value
    for label in unique(y)
        mask = y .== label
        cnt = sum(mask)
        
        if cnt > 1
            k = min(n_neighbors, cnt - 1)
            x_subset = x[mask]
            
            # build kd-tree and find distances
            tree = NearestNeighbors.KDTree(x_subset', NearestNeighbors.Euclidean())
            _, dists = NearestNeighbors.knn(tree, x_subset', k + 1, true)
            
            # store radius as distance to k-th neighbor
            r = [dists[i][end] for i in 1:cnt]
            
            radius[mask] = nextfloat.(r)
            k_all[mask] .= k
        end
        
        label_counts[mask] .= cnt
    end

    # ignore points with unique labels
    mask = label_counts .> 1
    n_valid_samples = sum(mask)
    
    # skip calculation if we don't have enough valid samples
    n_valid_samples == 0 && return 0.0
    
    label_counts = label_counts[mask]
    k_all        = k_all[mask]
    x_valid      = x[mask]
    radius_valid = radius[mask]
    
    # cnt points within radius in the continuous space
    tree_c = NearestNeighbors.KDTree(x_valid', NearestNeighbors.Euclidean())
    m_all = [length(NearestNeighbors.inrange(tree_c, [x_valid[i]], radius_valid[i])) - 1 for i in 1:n_valid_samples]

    # compute MI using Ross formula
    mi = digamma(n_valid_samples) + mean(digamma.(k_all)) - 
         mean(digamma.(label_counts)) - mean(digamma.(m_all))
    
    return max(0.0, mi)
end

# standardize a dataset along any axis
# center to the mean and component wise scale to unit variance
function scale!(
    X::AbstractArray{T};
    dims::Int64  = 1,
    center::Bool = true,
    scale::Bool  = true
) where {T<:Float64}
    if center
        mean_vals = StatsBase.mean(X, dims=dims)
        X .-= mean_vals
    end

    if scale
        std_vals = StatsBase.std(X, dims=dims, corrected=false)
        std_vals[std_vals .== 0] .= 1.0
        return X ./= std_vals
    end

    return X
end
scale!(X::AbstractArray{<:Real}; kwargs...) = scale!(Float64.(X); kwargs...)

# compute the digamma function of `x` (the logarithmic derivative of `gamma(x)`).
function digamma(x::Float64)
    # Taken from SpecialFunctions.jl package
    # The MIT License (MIT)
    # Copyright (c) 2017 Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and others:
    # https://github.com/JuliaMath/SpecialFunctions.jl/graphs/contributors
    if x ≤ 0 # reflection formula
        ψ = -π / tanpi(x)
        x = 1 - x
    else
        ψ = zero(x)
    end
    X = 8
    if x < X
        # shift using recurrence formula
        n = X - floor(Int,x)
        for ν = 1:n-1
            ψ -= inv(x + ν)
        end
        ψ -= inv(x)
        x += n
    end
    t = inv(x)
    ψ += log(x) - 0.5*t
    t *= t # 1/x^2
    # the coefficients here are Float64(bernoulli[2:9] .// (2*(1:8)))
    ψ -= t * @evalpoly(t,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686)
end
digamma(x::Real) = digamma(Float64(x))

mutual_info_classifier(X,y; n_neighbors=16, rng=Xoshiro(11))
