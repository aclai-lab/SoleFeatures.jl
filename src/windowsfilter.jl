struct MovingWindows
    nwindows::Int64
    reloverlap::Float64

    function MovingWindows(nwindows::Integer, reloverlap::AbstractFloat)
        nwindows <= 0 && throw(DomainError(nwindows, "Must be greater than 0"))
        !(0.0 <= reloverlap <= 1.0) &&
            throw(DomainError(reloverlap, "Must be within 0.0 and 1.0"))
        return new(nwindows, reloverlap)
    end
end

struct MovingWindowsIndex
    i::Int64
    mw::Ref{MovingWindows}

    function MovingWindowsIndex(i::Integer, mw::MovingWindows)
        i > mw.nwindows && throw(DomainError(i, ""))
        new(i, Ref(mw))
    end
end

struct Measure
    name::Symbol
    funct::Function
end

struct WindowsFilter{T <: AbstractFilterLimiter} <: AbstractWindowsFilter
    limiter::T
    # parameters
    usedselector::AbstractFeaturesSelector{AbstractLimiter}
    expansion::Vector{Tuple{Symbol, MovingWindows, Measure}}
    groupby::Vector{Union{Symbol, Tuple{Symbol, Symbol}}} # {:Attributes, :Windows, :Measures}

    function WindowsFilter(
        limiter::T,
        usedselector::AbstractFeaturesSelector{AbstractLimiter},
        expansion::Vector{Tuple{Symbol, MovingWindows, Measure}},
        groupby::Vector{Union{Symbol, Tuple{Symbol, Symbol}}}
    )
        return new(limiter, usedselector, expansion, groupby)
    end

    function WindowsFilter(
        limiter::T,
        usedselector::AbstractFeaturesSelector{AbstractLimiter},
        nwindows::Int64,
        reloverlap::Float64,
        measures::Vector{Measure}
        groupby::Vector{Symbol}
    )
        mw = MovingWindows(nwindows, reloverlap)
        expand_attrs = [ (ALL_ATTRIBUTES, mw, m) for m in measures]
        return new(limiter, usedselector, expand_attrs, groupby)
    end
end

const ALL_ATTRIBUTES = :*

# group section

const GROUP_ATTRIBUTES = :Attributes
const GROUP_WINDOWS = :Windows
const GROUP_MEASURES = :Measures

function _groupbyattributes() end
function _groupbywindows() end
function _groupbymeasures() end

const _GROUPDICT = Dict{Symbol, Function}(
    GROUP_ATTRIBUTES => _groupbyattributes,
    GROUP_WINDOWS => _groupbywindows,
    GROUP_MEASURES => _groupbymeasures
)

# getter

limiter(selector::WindowsFilter) = selector.limiter
expansion(selector::WindowsFilter) = selector.expansion
groupby(selector::WindowsFilter) = selector.groupby

function WindowsFittest(
    suiteness::AbstractFloat,
    selector::AbstractFeaturesSelector{AbstractLimiter},
    nwindows::Integer,
    relative_overlap::AbstractFloat
)
    return WindowsFilter(
        FittestLimiter(suiteness),
        selector,
        nwindows,
        relative_overlap
    )
end

function apply(
    df::AbstractDataFrame,
    selector::WindowsFilter{FittestLimiter}
)::Vector{Integer}

end

function gendataset(df::AbstractDataFrame, selector::WindowsFilter)::DataFrame
    ndf = DataFrame()
    expmode = expansion(selector)

    attrs = all(x->(x[1] === ALL_ATTRIBUTES), )

    group = groupby(selector)


    for colname in Symbol.(names(df))

    end

    return ndf
end
