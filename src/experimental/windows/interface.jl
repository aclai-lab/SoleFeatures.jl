abstract type AbstractMovingWindows end
abstract type AbstractMovingWindowsIndex end

function Base.length(mw::AbstractMovingWindows)
    return error("Not implemented for $(typeof(mw))")
end

function getwindows(mw::AbstractMovingWindows, np::Integer)
    return error("Not implemented for $(typeof(mw))")
end

# function getwindow(mw::AbstractMovingWindows, np::Integer, i::Integer)
#     return windows(mw, np)[i]
# end

function nwindows(mw::AbstractMovingWindows, np::Integer)
    return error("Not implemented for $(typeof(mw))")
end
