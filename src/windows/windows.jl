abstract type AbstractMovingWindows end
abstract type AbstractMovingWindowsIndex end

# moving windows index

struct MovingWindowsIndex{T <: AbstractMovingWindows} <: AbstractMovingWindowsIndex
    index::Int
    movingwindows::Base.RefValue{T}

    function MovingWindowsIndex(
        index::Integer,
        movingwindows::Base.Ref{T}
    ) where {T <: AbstractMovingWindows}
        index > length(movingwindows[]) &&
            throw(DimensionMismatch("Not valid index"))
        return new{T}(index, movingwindows)
    end
end

index(mwi::MovingWindowsIndex) = mwi.index
movingwindows(mwi::MovingWindowsIndex) = mwi.movingwindows[]

function Base.show(io::IO, ::MIME"text/plain", mwi::AbstractMovingWindowsIndex)
    print(io, "$(typeof(mwi))\n\t$(string(mwi))")
end

function Base.string(mwi::AbstractMovingWindowsIndex)
    mw = movingwindows(mwi)
    fv = join([ getfield(mw, i) for i in 1:nfields(mw) ], ",")
    return "W$(index(mwi))($(fv))"
end

function Base.isequal(mwi1::AbstractMovingWindowsIndex, mwi2::AbstractMovingWindowsIndex)
    return index(mwi1) == index(mwi2) && movingwindows(mwi1) == movingwindows(mwi2)
end

function getwindow(v::AbstractVector, mwi::AbstractMovingWindowsIndex)
    return getwindow(v, movingwindows(mwi), index(mwi))
end

# AbstractMovingWindow functions

function Base.getindex(mw::AbstractMovingWindows, i::Integer)
    return MovingWindowsIndex(i, Ref(mw))
end

function Base.isempty(mw::AbstractMovingWindows)
    return length(mw) == 0
end

function Base.firstindex(mw::AbstractMovingWindows)
    getindex(mw, 1)
end

function Base.lastindex(mw::AbstractMovingWindows)
    getindex(mw, length(mw))
end

function Base.iterate(mw::AbstractMovingWindows, i::Integer=1)
    i > length(mw) && return nothing
    return (getindex(mw, i), i+1)
end

function getwindow(v::AbstractVector, mw::AbstractMovingWindows, i::Integer)
    return getwindows(v, mw)[i]
end

# Fixed number moving windows

struct FixedNumMovingWindows <: AbstractMovingWindows
    nwindows::Int
    reloverlap::Float64

    function FixedNumMovingWindows(nwindows::Integer, reloverlap::AbstractFloat)
        nwindows <= 0 && throw(DomainError(nwindows, "Must be greater than 0"))
        !(0.0 <= reloverlap <= 1.0) &&
            throw(DomainError(reloverlap, "Must be within 0.0 and 1.0"))
        return new(nwindows, reloverlap)
    end
end

nwindows(mw::FixedNumMovingWindows) = mw.nwindows
reloverlap(mw::FixedNumMovingWindows) = mw.reloverlap

function Base.length(mw::FixedNumMovingWindows)
    nwindows(mw)
end

function Base.isequal(mw1::FixedNumMovingWindows, mw2::FixedNumMovingWindows)
    return nwindows(mw1) == nwindows(mw2) && reloverlap(mw1) == reloverlap(mn2)
end

function getwindows(v::AbstractVector, mw::FixedNumMovingWindows)
    return _moving_window(v; nwindows=nwindows(mw), relative_overlap=reloverlap(mw))
end

# Fixed size moving windows

mutable struct FixedSizeMovingWindows <: AbstractMovingWindows
    wsize::Int
    wstep::Int
    npoints::Union{Int, Nothing}

    FixedSizeMovingWindows(wsize::Integer, wstep::Integer) = new(wsize, wstep, nothing)
end

wsize(mw::FixedSizeMovingWindows) = mw.wsize
wstep(mw::FixedSizeMovingWindows) = mw.wstep
npoints(mw::FixedSizeMovingWindows) = mw.npoints
npoints!(mw::FixedSizeMovingWindows, n::Integer) = (mw.npoints = n)
npoints!(mw::FixedSizeMovingWindows, v::AbstractVector) = npoints!(mw, length(v))

# TODO: fix Base.length function
# function Base.length(mw::FixedSizeMovingWindows)
#     isnothing(npoints(md)) && throw(ErrorException("'npoints' not yet defined"))
#     @warn("This function doesn't return correct value :(")
#     return ceil(Integer, (npoints(mw) - wsize(mw) - 1) / wstep(mw))
# end

function Base.isequal(mw1::FixedSizeMovingWindows, mw2::FixedSizeMovingWindows)
    return wsize(mw1) == wsize(mw2) &&
        wstep(mw1) == wstep(mw2) &&
        npoints(mw1) == npoints(mw2)
end

function getwindows(v::AbstractVector, mw::FixedSizeMovingWindows)
    npoints!(mw, v)
    return _moving_window(v; window_size=wsize(mw), window_step=wstep(mw))
end
