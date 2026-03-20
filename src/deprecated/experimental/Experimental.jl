module Experimental

using DataFrames
using OrderedCollections

# ========================================================================================
# EXTRACTOR
# ========================================================================================

include("extraction.jl")

# ========================================================================================
# WINDOWS
# ========================================================================================

include("windows/interface.jl")
include("windows/data-filters.jl")
include("windows/windows.jl")

end # module
