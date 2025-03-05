# ---------------------------------------------------------------------------- #
#                               abstract types                                 #
# ---------------------------------------------------------------------------- #
"""
Abstract type for score struct
"""
abstract type AbstractScore end

"""
Abstract type for features selector result.
"""
abstract type AbstractSelResult end

# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const ABT = Union{NamedTuple{(:aggrby,:aggregatef,:group_before_score)}, Nothing}

# ---------------------------------------------------------------------------- #
#                                data structures                               #
# ---------------------------------------------------------------------------- #
"""
    Score <: AbstractScore

A struct representing the score of an individual feature in feature selection.

# Fields
- `id :: Int` : The unique identifier of the feature that this score belongs to
- `score :: Float64` : The numerical score value indicating feature importance/relevance

# Constructors
```julia
Score(id::Int, score::Float64)
"""
struct Score <: AbstractScore
    id    :: Int
    score :: Float64
end

# Value access methods
Base.getproperty(sc::Score, s::Symbol) = getfield(sc, s)
Base.propertynames(::Score)            = (:id, :score)

score_id(s::Score)  = s.id
score_val(s::Score) = s.score

"""
    GroupScore <: AbstractScore

A struct representing the score of a group of features in grouped feature selection.

# Fields
- `grp :: Tuple{Vararg{Symbol}}` : Tuple of symbols identifying the feature group
- `score :: Float64` : Numerical score value indicating the group's importance/relevance

# Constructors
```julia
GroupScore(grp::Tuple{Vararg{Symbol}}, score::Float64)
"""
struct GroupScore <: AbstractScore
    grp   :: Tuple{Vararg{Symbol}}
    score :: Float64
end

# Value access methods
Base.getproperty(sc::GroupScore, s::Symbol) = getfield(sc, s)
Base.propertynames(::GroupScore)            = (:id, :score)

score_id(s::GroupScore)  = s.id
score_val(s::GroupScore) = s.score

# )::Tuple{Vector{Int},Vector{Score}}
# )::Tuple{Vector{Int},Vector{Vector{Int}},Vector{GroupScore},Vector{Vector{<:Real}}}