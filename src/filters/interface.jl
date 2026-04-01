# ---------------------------------------------------------------------------- #
#                           filters abstract types                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractFilter{F,T,L,D} <: AbstractFeaturesSelector end
abstract type AbstractFilterInfo end

abstract type AbstractTask           end
abstract type AbstractLearning       end
abstract type AbstractDimensionality end

abstract type ClassificationTask <: AbstractTask end
abstract type RegressionTask     <: AbstractTask end

abstract type Supervised   <: AbstractLearning end
abstract type Unsupervised <: AbstractLearning end

abstract type Univariate   <: AbstractDimensionality end
abstract type Multivariate <: AbstractDimensionality end

Base.eltype(::AbstractFilter{F,T,L,D})        where {F,T,L,D} = F
get_task(::AbstractFilter{F,T,L,D})           where {F,T,L,D} = T
get_learning(::AbstractFilter{F,T,L,D})       where {F,T,L,D} = L
get_dimensionality(::AbstractFilter{F,T,L,D}) where {F,T,L,D} = D

get_rank(f::AbstractFilter{F,T,L,D})          where {F,T,L,D<:Univariate} = f.rank
get_score(f::AbstractFilter{F,T,L,D})         where {F,T,L,D<:Univariate} = f.score

function Base.show(io::IO, filter::AbstractFilter{F,T,L,D}) where {F,T,L,D}    
    n_features = length(filter.rank)
    top_n      = min(3, n_features)
    max_idx    = maximum(filter.rank[1:top_n])
    pad_width  = length(string(max_idx))
    
    println(io, typeof(filter))
    println(io, "  Features: $n_features")
    println(io, "  Top $top_n features (rank → score):")
    for i in 1:top_n
        feat_idx   = filter.rank[i]
        score_val  = filter.score[feat_idx]
        padded_idx = lpad(feat_idx, pad_width)
        println(io, "    $i. Feature $padded_idx → $(round(score_val, digits=4))")
    end
end

# # ---------------------------------------------------------------------------- #
# #                            functions definitions                             #
# # ---------------------------------------------------------------------------- #
# function limiter(selector::AbstractUnivariateFilterBased)
#     !hasproperty(selector, :limiter) &&
#         throw(ErrorException("`selector` struct not contain `limiter` field"))
#     return selector.limiter
# end

# function apply(
#     X::AbstractDataFrame,
#     selector::AbstractUnivariateFilterBased
# )
#     return limit(score(X, selector), limiter(selector))
# end

# function apply(
#     X::AbstractDataFrame,
#     y::AbstractVector{<:SoleData.SoleBase.CLabel},
#     selector::AbstractUnivariateFilterBased
# )
#     return limit(score(X, y, selector), limiter(selector))
# end
