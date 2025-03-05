# ---------------------------------------------------------------------------- #
#                                   group id                                   #
# ---------------------------------------------------------------------------- #
"""
    group_id_by_aggrby(Xinfo::AbstractVector{<:InfoFeat}, aggrby::Tuple{Vararg{Symbol}})::Vector{Vector{Int}}

Group indices of `InfoFeat` objects based on specified fields.

# Arguments
- `Xinfo::AbstractVector{<:InfoFeat}`: Vector of `InfoFeat` objects to be grouped
- `aggrby::Tuple{Vararg{Symbol}}`: Tuple of field names (as symbols) to group by

# Returns
- `Vector{Vector{Int}}`: A vector where each inner vector contains the indices of elements 
  that share the same values for the specified fields

# Description
This function groups elements in `Xinfo` based on the values of the fields specified in `aggrby`.
It first identifies all unique combinations of values for the specified fields, then creates
groups by finding all elements that match each unique combination.

# Throws
- `ErrorException`: If any resulting group is empty
"""
function group_id_by_aggrby(
    Xinfo::AbstractVector{<:InfoFeat},
    aggrby::Tuple{Vararg{Symbol}}
)::Vector{Vector{Int}}
    # get all unique combinations of values for the specified fields
    value_combinations = unique([
        Tuple(getfield(info, field) for field in aggrby)
        for info in Xinfo
    ])
    # for each unique combination, find all indices with matching values
    ixs = [
        findall(i -> Tuple(getfield(Xinfo[i], field) for field in aggrby) == combination, 
                1:length(Xinfo))
        for combination in value_combinations
    ]

    any(isempty.(ixs)) && throw(ErrorException("Some of the groups are empty!"))

    return ixs
end

# ---------------------------------------------------------------------------- #
#                             feature selection                                #
# ---------------------------------------------------------------------------- #
"""
    _fs(X, [y,] selector, limiter)

Perform a feature selection using `selector` limiting the variables selected
using `limiter`. `X` is the dataset as `AbstractDataFrame`.

If a supervised selector is passed the `y` parameter is needed: an `AbstractVector`
of labels.
"""
function _fs(
    X::AbstractMatrix,
    y::Union{AbstractVector,Nothing},
    Xinfo::AbstractVector{<:InfoFeat},
    selector::AbstractFeaturesSelector,
    limiter::AbstractLimiter
)::Tuple{Vector{Int},Vector{Score}}
    scores = isnothing(y) || is_unsupervised(selector) ? score(X, selector) : score(X, y, selector)
    idxes = SoleFeatures.limit(scores, limiter)

    return idxes, [SoleFeatures.Score(i.id, score) for (i, score) in zip(Xinfo, scores)]
end

function _fs(
    X::AbstractMatrix,
    Xinfo::AbstractVector{<:InfoFeat},
    selector::AbstractFeaturesSelector,
    limiter::AbstractLimiter
)::Tuple{Vector{Int},Vector{Score}}
    return _fs(X, nothing, Xinfo, selector, limiter)
end

_fs(Xdf::AbstractDataFrame, args...)::Tuple{Vector{Int},Vector{Score}} = _fs(Matrix(Xdf), args...)

# ---------------------------------------------------------------------------- #
#                          grouped feature selection                           #
# ---------------------------------------------------------------------------- #
"""
Perform feature selection on groups

## PARAMS

- `X`: the dataset in the form of `AbstractDataFrame`;
- `y`: the labels if the dataset is supervised; if the passed `selector` is supervised it has to be different from `nothing`;
- `selector`: the feature selection algorithm of type `AbstractFeaturesSelector`;
- `limiter`: the policy used to select the features to of type `AbstractLimiter`;
- `aggrby`: it is a tuple describing the portion of the column name to use to determine groups;
- `groups_separator`: the substring used to split the `DataFrame` names; default value is `"@@@"`.
- `aggregatef`: use this function to aggregate results from groups; default value is `identity`;
- `group_before_score`: it the passed `AbstractFeaturesSelector` is multivariate it can lead to
    different results to calculate scores after or before grouping variables by `aggrby` parameter.

## RETURN
return sel_idxes, g_indices, groups_score, scores

- first element: index of the selected groups
- second element: indices of variables for each group
- third element: score of each group
- fourth element: score of each variable grouped

# TODO: expand documentation
"""
function _fsgroup(
    X::AbstractMatrix,
    y::Union{Class,Nothing},
    Xinfo::AbstractVector{<:InfoFeat},
    selector::AbstractFeaturesSelector,
    limiter::AbstractLimiter,
    aggrby::Tuple{Vararg{Symbol}};
    aggregatef::Function=mean,
    group_before_score::Bool=true,
)::Tuple{Vector{Int},Vector{GroupScore},Vector{Vector{Int}},Vector{Vector{<:Real}}}
    g_indices = group_id_by_aggrby(Xinfo, aggrby)

    scores = []
    groups_score = GroupScore[]

    if group_before_score
        # group and then evaluate score internally to each group
        for g in g_indices

            s = isnothing(y) || is_unsupervised(selector) ? score(X[:,g], selector) : score(X[:,g], y, selector)

            push!(scores, s) # save scores of variables of current group
            grp = Tuple(Symbol.(collect(getfield(first(Xinfo[g]), a) for a in aggrby)))
            push!(groups_score, GroupScore(grp, aggregatef(s))) # save scores of variables of current group
        end
    else
        # calculate scores for all variables and then group
        allscores = isnothing(y) || is_unsupervised(selector) ? score(X, selector) : score(X, y, selector)

        for (i, cur_g_indices) in enumerate(g_indices)
            push!(scores, allscores[cur_g_indices]) # save scores of variables of current group
            grp = Tuple(Symbol.(collect(getfield(first(Xinfo[cur_g_indices]), a) for a in aggrby)))
            push!(groups_score, SoleFeatures.GroupScore(grp, aggregatef(allscores[cur_g_indices]))) # save aggregated group score
        end
    end

    # apply limiter on groups
    sel_idxes = SoleFeatures.limit(groups_score, limiter)

    return sel_idxes, groups_score, g_indices, scores
end

function _fsgroup(
    X::AbstractMatrix,
    Xinfo::AbstractVector{<:InfoFeat},
    selector::AbstractFeaturesSelector,
    limiter::AbstractLimiter,
    aggrby::Tuple{Vararg{Symbol}};
    kwargs...
)::Tuple{Vector{Int},Vector{GroupScore},Vector{Vector{Int}},Vector{Vector{<:Real}}}
    return _fsgroup(X, nothing, Xinfo, selector, limiter, aggrby; kwargs...)
end

_fsgroup(Xdf::AbstractDataFrame, args...) = _fsgroup(Matrix(Xdf), args...)