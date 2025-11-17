using SparseArrays
using LinearAlgebra
using Distances

"""
    construct_W(X; kwargs...)

Construct the affinity matrix W through different ways.

# Arguments
- `X::AbstractMatrix`: Input data of shape (n_samples, n_features)
- `y::AbstractVector`: True label information (required for supervised mode)
- `metric::String="cosine"`: Distance measure ("euclidean" or "cosine")
- `neighbor_mode::String="knn"`: Graph construction mode ("knn" or "supervised")
- `weight_mode::String="binary"`: Edge weighting ("binary", "heat_kernel", or "cosine")
- `k::Int=5`: Number of neighbors
- `t::Float64=1.0`: Parameter for heat_kernel weight mode
- `fisher_score::Bool=false`: Build affinity matrix in fisher score way
- `reliefF::Bool=false`: Build affinity matrix in reliefF way

# Returns
- `W::SparseMatrixCSC`: Sparse affinity matrix of shape (n_samples, n_samples)
"""
function construct_W(X::AbstractMatrix; 
                     metric::String="cosine",
                     neighbor_mode::String="knn",
                     weight_mode::String="binary",
                     k::Int=5,
                     t::Float64=1.0,
                     fisher_score::Bool=false,
                     reliefF::Bool=false,
                     y::Union{AbstractVector,Nothing}=nothing)
    
    n_samples, n_features = size(X)
    
    # Validate supervised mode requirements
    if neighbor_mode == "supervised" && isnothing(y)
        error("Label y is required in supervised neighbor mode!")
    end
    
    # Adjust metric for specific weight modes
    if weight_mode == "heat_kernel"
        metric = "euclidean"
    elseif weight_mode == "cosine"
        metric = "cosine"
    end
    
    # KNN neighbor mode
    if neighbor_mode == "knn"
        return _construct_W_knn(X, k, metric, weight_mode, t)
    
    # Supervised neighbor mode
    elseif neighbor_mode == "supervised"
        if fisher_score
            return _construct_W_fisher(X, y)
        elseif reliefF
            return _construct_W_reliefF(X, y, k)
        else
            return _construct_W_supervised(X, y, k, metric, weight_mode, t)
        end
    end
    
    error("Unknown neighbor_mode: $neighbor_mode")
end

# Helper: KNN mode
function _construct_W_knn(X, k, metric, weight_mode, t)
    n_samples = size(X, 1)
    
    if metric == "euclidean"
        D = pairwise(Euclidean(), X', X', dims=2)
        D .= D .^ 2
    elseif metric == "cosine"
        X_norm = X ./ max.(sqrt.(sum(X .^ 2, dims=2)), 1e-12)
        D = X_norm * X_norm'
        D = -D  # For sorting (higher cosine = closer)
    end
    
    # Get k+1 nearest neighbors
    idx = [sortperm(D[i, :], rev=(metric=="cosine"))[1:k+1] for i in 1:n_samples]
    
    # Build triplets
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    
    for i in 1:n_samples
        for j in idx[i]
            push!(I_vec, i)
            push!(J_vec, j)
            
            if weight_mode == "binary"
                push!(V_vec, 1.0)
            elseif weight_mode == "heat_kernel"
                push!(V_vec, exp(-D[i, j] / (2 * t * t)))
            elseif weight_mode == "cosine"
                push!(V_vec, -D[i, j])  # Undo negation
            end
        end
    end
    
    W = sparse(I_vec, J_vec, V_vec, n_samples, n_samples)
    
    # Symmetrize
    bigger = W' .> W
    W = W - W .* bigger + W' .* bigger
    
    return W
end

# Helper: Fisher score mode
function _construct_W_fisher(X, y)
    n_samples = size(X, 1)
    labels = unique(y)
    n_classes = length(labels)
    
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    
    for label in labels
        class_idx = findall(==(label), y)
        n_class = length(class_idx)
        weight = 1.0 / n_class
        
        for i in class_idx, j in class_idx
            push!(I_vec, i)
            push!(J_vec, j)
            push!(V_vec, weight)
        end
    end
    
    return sparse(I_vec, J_vec, V_vec, n_samples, n_samples)
end

# Helper: ReliefF mode
function _construct_W_reliefF(X, y, k)
    n_samples = size(X, 1)
    labels = unique(y)
    n_classes = length(labels)
    
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    
    # NH(x): same class neighbors
    for label in labels
        class_idx = findall(==(label), y)
        if length(class_idx) <= 1
            continue
        end
        
        X_class = X[class_idx, :]
        D = pairwise(Euclidean(), X_class', X_class', dims=2) .^ 2
        
        k_eff = min(k, length(class_idx) - 1)
        
        for (i_local, i_global) in enumerate(class_idx)
            neighbors = sortperm(D[i_local, :])[1:k_eff+1]
            
            for j_local in neighbors
                j_global = class_idx[j_local]
                push!(I_vec, i_global)
                push!(J_vec, j_global)
                push!(V_vec, i_global == j_global ? 1.0 : 1.0 / k_eff)
            end
        end
    end
    
    # NM(x,y): different class neighbors
    for label_i in labels
        class_idx_i = findall(==(label_i), y)
        X_i = X[class_idx_i, :]
        
        for label_j in labels
            if label_i == label_j
                continue
            end
            
            class_idx_j = findall(==(label_j), y)
            X_j = X[class_idx_j, :]
            
            D = pairwise(Euclidean(), X_i', X_j', dims=2)
            
            for (i_local, i_global) in enumerate(class_idx_i)
                neighbors = sortperm(D[i_local, :])[1:k]
                
                for j_local in neighbors
                    j_global = class_idx_j[j_local]
                    push!(I_vec, i_global)
                    push!(J_vec, j_global)
                    push!(V_vec, -1.0 / ((n_classes - 1) * k))
                end
            end
        end
    end
    
    W = sparse(I_vec, J_vec, V_vec, n_samples, n_samples)
    
    # Symmetrize negative part
    bigger = W' .> W
    W = W - W .* bigger + W' .* bigger
    
    return W
end

# Helper: Supervised mode (general)
function _construct_W_supervised(X, y, k, metric, weight_mode, t)
    n_samples = size(X, 1)
    labels = unique(y)
    
    I_vec = Int[]
    J_vec = Int[]
    V_vec = Float64[]
    
    for label in labels
        class_idx = findall(==(label), y)
        X_class = X[class_idx, :]
        
        if metric == "euclidean"
            D = pairwise(Euclidean(), X_class', X_class', dims=2) .^ 2
        elseif metric == "cosine"
            X_norm = X_class ./ max.(sqrt.(sum(X_class .^ 2, dims=2)), 1e-12)
            D = -(X_norm * X_norm')
        end
        
        for (i_local, i_global) in enumerate(class_idx)
            neighbors = sortperm(D[i_local, :], rev=(metric=="cosine"))[1:k+1]
            
            for j_local in neighbors
                j_global = class_idx[j_local]
                push!(I_vec, i_global)
                push!(J_vec, j_global)
                
                if weight_mode == "binary"
                    push!(V_vec, 1.0)
                elseif weight_mode == "heat_kernel"
                    push!(V_vec, exp(-D[i_local, j_local] / (2 * t * t)))
                elseif weight_mode == "cosine"
                    push!(V_vec, -D[i_local, j_local])
                end
            end
        end
    end
    
    W = sparse(I_vec, J_vec, V_vec, n_samples, n_samples)
    
    # Symmetrize
    bigger = W' .> W
    W = W - W .* bigger + W' .* bigger
    
    return W
end