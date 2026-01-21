# ---------------------------------------------------------------------------- #
#                           filters abstract types                             #
# ---------------------------------------------------------------------------- #
abstract type AbstractFilter{F,T,L,D} end
abstract type AbstractFilterInfo      end

abstract type AbstractTask            end
abstract type AbstractLearning        end
abstract type AbstractDimensionality  end

abstract type ClassificationTask <: AbstractTask     end
abstract type RegressionTask     <: AbstractTask     end

abstract type Supervised   <: AbstractLearning       end
abstract type Unsupervised <: AbstractLearning       end

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

# ---------------------------------------------------------------------------- #
#                           limiters abstract types                            #
# ---------------------------------------------------------------------------- #
abstract type AbstractLimiter{F,T,L,D} end
abstract type AbstractLimiterInfo end

Base.eltype(::AbstractLimiter{F,T,L,D})        where {F,T,L,D} = F
get_task(::AbstractLimiter{F,T,L,D})           where {F,T,L,D} = T
get_learning(::AbstractLimiter{F,T,L,D})       where {F,T,L,D} = L
get_dimensionality(::AbstractLimiter{F,T,L,D}) where {F,T,L,D} = D

get_filter(l::AbstractLimiter{F,T,L,D})        where {F,T,L,D} = l.filter
get_rank(l::AbstractLimiter{F,T,L,D})          where {F,T,L,D} = l.rank
get_info(l::AbstractLimiter{F,T,L,D})          where {F,T,L,D} = l.info

function Base.show(io::IO, limiter::AbstractLimiter{F,T,L,D}) where {F,T,L,D}
    println(io, typeof(limiter))
    has_rank = hasproperty(limiter, :rank)
    has_filter = hasproperty(limiter, :filter)
    scores = has_filter ? get_score(getproperty(limiter, :filter)) : nothing

    if has_rank
        rank = getproperty(limiter, :rank)
        n = length(rank)
        top_n = min(3, n)
        println(io, "  Selected: $n")
        if scores !== nothing
            max_idx = maximum(rank[1:top_n])
            pad_width = length(string(max_idx))
            println(io, "  Top $top_n (rank → score):")
            for i in 1:top_n
                idx = rank[i]
                sv = scores[idx]
                pid = lpad(idx, pad_width)
                println(io, "    $i. Feature $pid → $(round(sv, digits=4))")
            end
        else
            println(io, "  Top $top_n indices:")
            for i in 1:top_n
                println(io, "    $i. $(rank[i])")
            end
        end
    end
    if hasproperty(limiter, :info)
        println(io, "  Info:")
        show(io, limiter.info)
    end
end
