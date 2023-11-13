struct PearsonCorFilter{T <: AbstractLimiter} <: AbstractPearsonCorFilter{T}
    limiter::T
    # parameters
end

# ========================================================================================
# TRAITS

is_supervised(::AbstractPearsonCorFilter) = true

# ========================================================================================
# SCORE

function score(
    X::AbstractDataFrame,
    y::AbstractVector{<:Class},
    selector::PearsonCorFilter
)::Vector{Float64}
    coltypes = eltype.(eachcol(X))
    uncalcidxes = findall(==(false), coltypes .<: Real)
    if (!isempty(uncalcidxes))
        throw(DomainError("Columns must be subtype of Real.\n
                    The following column indices are not handable: $(uncalcidxes)"))
    end
    scores = cor.(eachcol(X), [y])
    return scores
end

# ========================================================================================
# CUSTOM CONSTRUCTORS

PearsonCorRanking(nbest) =  PearsonCorFilter(RankingLimiter(nbest, false))
