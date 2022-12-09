using OrderedCollections
using DataFrames
using StatsBase

# limiter
include("../src/limiter/interfaces.jl")
include("../src/limiter/functions.jl")
# fs
include("../src/interfaces.jl")
# windows
include("../src/windows/data-filters.jl")
include("../src/windows/windows.jl")
include("../src/windowsfilter/utils.jl")


df = DataFrame(:firstcol => [rand(4), rand(4), rand(4)],
                :secondcol => [rand(4), rand(4), rand(4)])
attrs = Symbol.(names(df))
fnmw = FixedNumMovingWindows(3, 0.25)
measures = [minimum, maximum]
awmds = build_awds(attrs, [ fnmw... ], measures);
expand(df, awmds)

include("./test_function.jl")

df = random_timeseries_df()
attrname = Symbol.(names(df))
movwin = FixedNumMovingWindows(3, 0.2)
measures = [ minimum, maximum, mean ]
expansions = [ Iterators.product(attrname, movwin, measures)... ]

ndf = expand(df, expansions)
