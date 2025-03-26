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

# ---------------------------------------------------------------------------- #
#                      main feature selection function                         #
# ---------------------------------------------------------------------------- #
"""
TODO: documentation

# Feature Selection with Aggregation Control

## Overview
The `feature_selection` function allows precise control over how feature aggregation
is applied during the multi-step feature selection process.

## Aggregation Parameter (`aggrby`)
The `aggrby` parameter can be provided in two ways:

1. **Single NamedTuple**: When provided as a single NamedTuple (not a vector), 
   aggregation is only applied during the final step of feature selection.
   The function automatically creates a vector where:
   - All positions except the last contain `nothing`
   - The last position contains the provided aggregation parameters

2. **Vector of NamedTuples**: When provided as a vector, each element specifies 
   the aggregation behavior for the corresponding step in `fs_methods`.
"""
function feature_selection(
    X::AbstractMatrix{T},
    y::Union{AbstractVector{<:Class}, Nothing},
    Xinfo::Vector{<:InfoFeat};

    aggrby::Union{ABT,AbstractVector{<:ABT}} = (
        aggrby = (:var,),
        aggregatef = length, # NOTE: or mean, minimum, maximum to aggregate scores instead of just counting number of selected features for each group
        group_before_score = true,
    ),

    fs_methods::AbstractVector{<:NamedTuple{(:selector, :limiter)}} = [
        ( # STEP 1: unsupervised variance-based filter
            selector = VarianceFilter(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
        ( # STEP 2: supervised Mutual Information filter
            selector = MutualInformationClassif(SoleFeatures.IdentityLimiter()),
            limiter = PercentageLimiter(0.1),
        ),
        ( # STEP 3: group results by variable
            selector = IdentityFilter(),
            limiter = IdentityLimiter(),
        ),
    ],

    norm::Bool = false,
    normalize_kwargs::NamedTuple = NamedTuple()
)::Tuple{DataFrame, Vector{InfoFeat}, Vector{NamedTuple}} where {T<:Number}
# ) where {T<:Number}
    # prepare aggregation parameters
    if !(aggrby isa AbstractVector)
        # when aggrby is not a Vector assume that the user want to perform aggregation
        #    only during the last step of feature selection TODO: document this properly!!!
        aggrby = push!(Union{Nothing,NamedTuple}[fill(nothing, max(length(fs_methods)-1, 0))...], aggrby)
    end

    # prepare labels
    y_coded = @. CategoricalArrays.levelcode(y)

    # dataset normalization
    norm && _normalize_dataset(X, Xinfo; normalize_kwargs...)

    # feature selection
    fs_mid_results = NamedTuple{(:score, :indices,:group_aggr_func,:group_indices,:aggrby)}[]

    for (fsm, gfs_params) in zip(fs_methods, aggrby)
        current_dataset_col_slice = 1:size(X, 2)

        # pick survived columns only
        for f in fs_mid_results
            current_dataset_col_slice = current_dataset_col_slice[f.indices]
        end

        currX = X[:,current_dataset_col_slice]
        currXinfo = Xinfo[current_dataset_col_slice]

        dataset_param = isnothing(y_coded) || SoleFeatures.is_unsupervised(fsm.selector) ? 
            (currX, currXinfo) : 
            (currX, y_coded, currXinfo)

        idxes, score, g_indices =
            if isnothing(gfs_params)
                # perform normal feature selection
                SoleFeatures._fs(dataset_param..., fsm...)..., nothing
            else
                # perform aggregated feature selection
                sel_g_indices, g_scores, g_indices, grouped_variable_scores = SoleFeatures._fsgroup(
                    dataset_param..., fsm..., gfs_params.aggrby;
                    aggregatef = gfs_params.aggregatef,
                    group_before_score = gfs_params.group_before_score
                )

                # find indices to re-sort the scores of all variables to their
                # original position in dataset columns
                old_sort = sortperm(vcat(g_indices...))

                vcat(g_indices[sel_g_indices]...), vcat(vcat(grouped_variable_scores...)[old_sort]...), g_indices
            end

        sort!(idxes)

        push!(fs_mid_results, (
            score = score,
            indices = idxes,
            group_aggr_func = isnothing(gfs_params) ? nothing : gfs_params.aggregatef,
            group_indices = g_indices,
            aggrby = isnothing(gfs_params) ? nothing : gfs_params.aggrby
        ))
    end
    
    dataset_col_slice = 1:size(X, 2)

    for f in fs_mid_results
        dataset_col_slice = dataset_col_slice[f.indices]
    end

    namecols = [string(Xinfo[d].feat) * "(" * Xinfo[d].var * ")w" * string(Xinfo[d].nwin) for d in dataset_col_slice]

    return DataFrame(X[:,dataset_col_slice], namecols), Xinfo[dataset_col_slice], fs_mid_results
end

feature_selection(
    Xdf::AbstractDataFrame, 
    args...; 
    kwargs...
)::Tuple{DataFrame, Vector{InfoFeat}, Vector{NamedTuple}} = feature_selection(Matrix(Xdf), args...; kwargs...)
