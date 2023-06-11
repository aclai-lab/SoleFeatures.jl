# ========================================================================================
# USE THIS TEMPLATE TO CREATE NEW FILTER SELECTOR
# ========================================================================================

struct CustomFilter{T <: AbstractLimiter} <: AbstractCustomFilter{T}
    limiter::T
    # parameters
end

# ========================================================================================
# ACCESSORS

# Here you can write your own accessors to the struct parameters.

# ========================================================================================
# TRAITS

# Here you can write the selector traits.
# Traits can be:
#   isunivariate(::AbstractCustomFilter) = true
#   issupervised(::AbstractCustomFilter) = true
#   ...

# ========================================================================================
# SCORES

# Here you can write your own scores function

# Remove this function if selector has only SUPERVISED implementation
function getscores(
    X::AbstractDataFrame,
    selector::CustomFilter{<:AbstractLimiter}
)
    scores = []
    return scores
end

# Remove this function if selector has only UNSUPERVISED implementation
function getscores(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::CustomFilter{<:AbstractLimiter}
)
    scores = []
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

# Here you can write your own selector constructors
# Example:
#     - CustomFilter with ranking limiter:
#           CustomRanking(nbest) = CustomFilter(RankingLimiter(nbest, true))
#
#     - CustomFilter with threshold limiter
#           CustomThreshold(threshold) = CustomFilter(ThresholdLimiter(threshold, >=))

# ========================================================================================
# UTILS
