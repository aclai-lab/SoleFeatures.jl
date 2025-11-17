using MLJ
using StatsBase
using CategoricalArrays
using HypothesisTests
using MLBase
using Distributions

X, y = @load_iris
X = MLJ.matrix(X) 
y = MLJ.levelcode.(y)

using SparseArrays
using LinearAlgebra

"""
This function implements the fisher score feature selection, steps are as follows:
1. Construct the affinity matrix W in fisher score way
2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1

Input
-----
X: Matrix of shape (n_samples, n_features) - input data
y: Vector of shape (n_samples,) - input class labels
mode: "rank" (default) returns scores, "index" returns sorted feature indices

Output
------
score: Vector of shape (n_features,) - fisher score for each feature

Reference
---------
He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
"""
function fisher_score(
    X::AbstractMatrix{T},
    y::AbstractVector
)::Vector{Int64} where {T<:Float64}
    weigths    = _construct_w_fisher(T, y, size(X, 1))
    degrees    = sum(weigths, dims=2)
    tot_degree = sum(degrees)

    tmp = degrees' * X
    t1  = X .* degrees
    t2  = weigths' * X

    within_class_var  = sum(t1 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree
    within_class_var  = ifelse.(within_class_var .< 1e-12, 1e4, within_class_var)
    between_class_var = sum(t2 .* X, dims=1) .- (tmp .* tmp) ./ tot_degree

    lap_score    = 1 .- (between_class_var ./ within_class_var)
    score = vec(1.0 ./ lap_score .- 1.0)
    return sortperm(score)
end

function _construct_w_fisher(
    ::Type{T},
    y::AbstractVector,
    n::Int64
) where {T<:Float64}
    weights = zeros(T, n, n)
    
    foreach(unique(y)) do label
        idx = findall(==(label), y)
        @inbounds @views weights[idx, idx] .= 1.0 / length(idx)
    end
    
    return weights
end