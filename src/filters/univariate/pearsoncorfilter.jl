# ---------------------------------------------------------------------------- #
#                               pearson filter                                 #
# ---------------------------------------------------------------------------- #
struct PearsonCorFilter{T <: AbstractLimiter} <: AbstractPearsonCorFilter{T}
    limiter::T
    # TODO parameters
end

is_supervised(::AbstractPearsonCorFilter) = true
is_unsupervised(::AbstractPearsonCorFilter) = false

function score(
    X::AbstractArray,
    y::Vector{Int64},
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

# ---------------------------------------------------------------------------- #
#                             custom constructors                              #
# ---------------------------------------------------------------------------- #
PearsonCorRanking(nbest) =  PearsonCorFilter(RankingLimiter(nbest, false))