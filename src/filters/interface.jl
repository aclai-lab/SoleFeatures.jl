# ---------------------------------------------------------------------------- #
#                           filters abstract types                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractFilterBased{F,T,L,D} end

abstract type AbstractTask           end
abstract type AbstractLearning       end
abstract type AbstractDimensionality end

abstract type ClassificationTask <: AbstractTask end
abstract type RegressionTask     <: AbstractTask end

abstract type Supervised   <: AbstractLearning end
abstract type Unsupervised <: AbstractLearning end

abstract type Univariate   <: AbstractDimensionality end
abstract type Multivariate <: AbstractDimensionality end

Base.eltype(::AbstractFilterBased{F,T,L,D})        where {F,T,L,D} = F
get_task(::AbstractFilterBased{F,T,L,D})           where {F,T,L,D} = T
get_learning(::AbstractFilterBased{F,T,L,D})       where {F,T,L,D} = L
get_dimensionality(::AbstractFilterBased{F,T,L,D}) where {F,T,L,D} = D

get_rank(f::AbstractFilterBased{F,T,L,D})          where {F,T,L,D<:Univariate} = f.rank
get_score(f::AbstractFilterBased{F,T,L,D})         where {F,T,L,D<:Univariate} = f.score

function Base.show(io::IO, filter::AbstractFilterBased{F,T,L,D}) where {F,T,L,D}    
    n_features = length(filter.rank)
    top_n      = min(5, n_features)
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
# function score(
#     X::AbstractDataFrame,
#     selector::AbstractUnivariateFilterBased{<:AbstractLimiter}
# )
#     return error("`score` for unsupervised selectors not implemented " *
#         "for type: $(typeof(selector))")
# end

# function score(
#     X::AbstractDataFrame,
#     y::AbstractVector{<:SoleData.SoleBase.CLabel},
#     selector::AbstractUnivariateFilterBased{<:AbstractLimiter}
# )
#     return error("`score` for supervised selectors not implemented " *
#         "for type: $(typeof(selector))")
# end

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
