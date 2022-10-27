# alias: Attribute, MovingWindows, Measure

const AWMDescriptor = Tuple{Symbol, AbstractMovingWindowsIndex, Function}

# constants

const GROUPBY_ATTRIBUTES = :Attributes
const GROUPBY_WINDOWS = :Windows
const GROUPBY_MEASURES = :Measures
const GROUPBY_ID_DICT = Dict{Symbol, Integer}(
    GROUPBY_ATTRIBUTES => 1,
    GROUPBY_WINDOWS => 2,
    GROUPBY_MEASURES => 3
)
const SEPARATOR = "@@"

isdefined(Main, :Catch22) && (nameof(f::SuperFeature) = getname(f)) # wrap for Catch22

function expand(
    df::AbstractDataFrame,
    expansions::AbstractVector{<:AWMDescriptor}
)::DataFrame
    # TODO: use @Threads.threads and use sort of AWMDescriptor
    ndf = DataFrame()
    for ex in expansions
        colname = ex[1]
        insertcols!(ndf, _awm2str(ex) => _buildcol(df[:, colname], ex))
    end
    return ndf
end

function _buildcol(col::AbstractVector, expansion::AWMDescriptor)
    movwin = expansion[2]
    measuref = expansion[3]
    return measuref.([ getwindow(row, movwin) for row in col ])
end

"""
    _awm2str(expansion)

return name to use in expanded DataFrames from AWMDescriptor
"""
function _awm2str(expansion::AWMDescriptor)::Symbol
    attrname = string(expansion[1])
    movwin = string(expansion[2])
    measuref = nameof(expansion[3])
    return Symbol(join([attrname, movwin, measuref], SEPARATOR))
end

function retrive_groups(
    attributes::AbstractVector{Symbol},
    groupby::Union{Symbol, Tuple{Symbol, Symbol}}
)::Vector{Vector{Symbol}}
    groups = Dict{String, Vector{Symbol}}()
    # from: https://discourse.julialang.org/t/how-can-i-access-multiple-values-of-a-dictionary-using-a-tuple-of-keys/56868/3
    groupids = collect(getindex.(Ref(GROUPBY_ID_DICT), groupby)) # collect(Tuple) -> Vector

    for attr in attributes
        # retrive indicated group 'groupids' from current attribute and check if it belongs to 'groups'
        sel = join(split(String(attr), SEPARATOR)[groupids])
        if (haskey(groups, sel))
            push!(groups[sel], attr)
        else
            groups[sel] = [ attr ]
        end
    end

    return [ values(groups)... ]
end

function retrive_groups(
    expansions::AbstractVector{<:AWMDescriptor},
    groupby::Union{Symbol, Tuple{Symbol, Symbol}},
)::Vector{Vector{AWMDescriptor}}
    groups = Dict{String, Vector{AWMDescriptor}}()
    # from: https://discourse.julialang.org/t/how-can-i-access-multiple-values-of-a-dictionary-using-a-tuple-of-keys/56868/3
    groupids = [ collect(getindex.(Ref(GROUPBY_ID_DICT), groupby))... ] # collect(Tuple) -> Vector
    for exs in expansions
        # retrive indicated group 'groupids' from current expansion and check if it belongs to 'groups'
        sel = string(exs[groupids]...)
        if (haskey(groups, sel))
            push!(groups[sel], exs)
        else
            groups[sel] = [ exs ]
        end
    end
    return [ values(groups)... ]
end
