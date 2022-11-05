struct WindowsFilter{T <: AbstractFilterLimiter} <: AbstractWindowsFilter{T}
    limiter::T
    # parameters
    usedselector::AbstractFeaturesSelector
    expansions::Vector{<:AWMDescriptor}
    groupby::Vector{Union{Symbol, Tuple{Symbol, Symbol}}}

    # function WindowsFilter( # simple constructor
    #     limiter::T,
    #     usedselector::AbstractFeaturesSelector{AbstractLimiter},
    #     expansions::Vector{<:AWMDescriptor},
    #     groupby::Vector{Union{Symbol, Tuple{Symbol, Symbol}}}
    # ) where {T <: AbstractFilterLimiter}
    #     !allunique(expansions) &&
    #         throw(DomainError(expansions, "Repeated expansion"))
    #     return new{T}(limiter, usedselector, expansions, groupby)
    # end

    function WindowsFilter( # most generic constructor
        limiter::T,
        usedselector::AbstractFeaturesSelector,
        attributes::Union{Symbol, AbstractVector{Symbol}},
        movingwindows_idxes::Union{MovingWindowsIndex{R}, AbstractVector{MovingWindowsIndex{R}}},
        measures::Union{Function, AbstractVector{Function}},
        groupby::Union{Symbol, Tuple{Symbol, Symbol}, AbstractVector{Union{Symbol, Tuple{Symbol, Symbol}}}}
    ) where {T <: AbstractFilterLimiter, R <: AbstractMovingWindows}
        !isa(attributes, Symbol) && !allunique(attributes) &&
            throw(DomainError(attributes, "Repeated attributes"))
        !allunique(movingwindows_idxes) &&
            throw(DomainError(movingwindows_idxes, "Repeated moving windows"))
        !allunique(measures) &&
            throw(DomainError(measures, "Repeated measures"))
        isa(attributes, Vector) && _ALL_ATTRIBUTES in attributes &&
            throw(DomainError(attributes, "Invalid symbol in vector: $(_ALL_ATTRIBUTES)"))

        attributes = isa(attributes, Symbol) ? [ attributes ] : attributes
        movingwindows_idxes = [ movingwindows_idxes... ]
        measures = [ measures... ]
        groupby = !isa(groupby, Vector) ? [ groupby ] : groupby

        expansions = [ Iterators.product(attributes, movingwindows_idxes, measures)... ]
        return new{T}(limiter, usedselector, expansions, groupby)
    end

    function WindowsFilter( # all moving windows of one type constructor
        limiter::T,
        usedselector::AbstractFeaturesSelector,
        attributes::Union{Symbol, AbstractVector{Symbol}},
        movingwindows::AbstractMovingWindows,
        measures::Union{Function, AbstractVector{Function}},
        groupby::Union{Symbol, Tuple{Symbol, Symbol}, AbstractVector{Union{Symbol, Tuple{Symbol, Symbol}}}}
    ) where {T <: AbstractFilterLimiter}
        return WindowsFilter(
            limiter,
            usedselector,
            attributes,
            [ movingwindows... ],
            measures,
            groupby
        )
    end

    function WindowsFilter( # all attributes, all moving windows of one type constructor
        limiter::T,
        usedselector::AbstractFeaturesSelector,
        movingwindows::AbstractMovingWindows,
        measures::Union{Function, AbstractVector{Function}},
        groupby::Union{Symbol, Tuple{Symbol, Symbol}, AbstractVector{Union{Symbol, Tuple{Symbol, Symbol}}}}
    ) where {T <: AbstractFilterLimiter}
        return WindowsFilter(
            limiter,
            usedselector,
            _ALL_ATTRIBUTES,
            [ movingwindows... ],
            measures,
            groupby
        )
    end

end

# group section

const _ALL_ATTRIBUTES = :*

# getter

usedselector(selector::WindowsFilter) = selector.usedselector
limiter(selector::WindowsFilter) = selector.limiter
expansions(selector::WindowsFilter) = selector.expansions
groupby(selector::WindowsFilter) = selector.groupby

"""
    expand(df, selector)

extensions will be changed if _ALL_ATTRIBUTES is used
"""
function expand(df::AbstractDataFrame, selector::WindowsFilter)
    exps = expansions(selector)
    length(exps) == 0 && throw(ErrorException("No expansions found"))
    if (exps[1][1] === _ALL_ATTRIBUTES)
        newexps = _expand_all_attributes(Symbol.(names(df)), exps)
        empty!(exps)
        push!(exps, newexps...)
    end
    return expand(df, exps)
end

function _expand_all_attributes(
    attrs::AbstractVector{Symbol},
    expansions::AbstractVector{<:AWMDescriptor}
)
    return [ (a, e[2], e[3]) for a in attrs for e in expansions ]
end

function evaluate(df::AbstractDataFrame, wf::WindowsFilter)::Vector{<:AWMDescriptor}
    return evaluate(
        expand(df, wf),
        expansions(wf),
        usedselector(wf),
        groupby(wf),
        limiter(wf);
        already_expanded=true
    )
end
