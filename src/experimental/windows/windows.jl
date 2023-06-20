# ======================== moving windows index

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

(i::MovingWindowsIndex)(m::AbstractArray) = getwindow(m, i)

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

function getwindow(a::AbstractArray, mwi::AbstractMovingWindowsIndex)
    return getwindow(a, movingwindows(mwi), index(mwi))
end

# ======================== AbstractMovingWindow functions

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

function Base.iterate(mw::AbstractMovingWindows, i::Integer = 1)
    i > length(mw) && return nothing
    return (getindex(mw, i), i+1)
end

function getwindow(v::AbstractArray, mw::AbstractMovingWindows, i::Integer)
    return getwindows(v, mw)[i]
end
function getwindow(v::AbstractArray, mw::AbstractMovingWindows, i::Integer...)
    # TODO: need to check length(i) == ndims(v)
    return getwindows(v, mw)[i...]
end
function getwindow(v::AbstractArray, mw::AbstractMovingWindows, i::AbstractVector{<:Integer})
    # TODO: need to check length(i) == ndims(v)
    return getwindows(v, mw)[i...]
end
function getwindow(v::AbstractArray{N}, mw::AbstractMovingWindows, i::NTuple{N,<:Integer}) where N
    return getwindows(v, mw)[i...]
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
    return nwindows(mw)
end

function Base.isequal(mw1::FixedNumMovingWindows, mw2::FixedNumMovingWindows)
    return nwindows(mw1) == nwindows(mw2) && reloverlap(mw1) == reloverlap(mn2)
end

function getwindows(v::AbstractVector, mw::FixedNumMovingWindows)
    return _moving_window(v; nwindows=nwindows(mw), relative_overlap=reloverlap(mw))
end
function getwindows(v::AbstractArray, mw::FixedNumMovingWindows)
    indices = Base.product(_moving_window.(range.(1, size(v)); nwindows=nwindows(mw), relative_overlap=reloverlap(mw))...)
    return [v[idxs...] for idxs in indices]
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
function getwindows(v::AbstractArray, mw::FixedSizeMovingWindows)
    indices = Base.product(_moving_window.(range.(1, size(v)); window_size=wsize(mw), window_step=wstep(mw))...)
    return [v[idxs...] for idxs in indices]
end


# ### Centered Window ###

# Fixed number moving windows

struct CenteredMovingWindow <: AbstractMovingWindows
    nwindows::Int

    function CenteredMovingWindow(nwindows::Integer)
        nwindows <= 0 && throw(DomainError(nwindows, "Must be greater than 0"))
        return new(nwindows)
    end
end

nwindows(mw::CenteredMovingWindow) = mw.nwindows

function Base.length(mw::CenteredMovingWindow)
    return nwindows(mw)
end

function Base.isequal(mw1::CenteredMovingWindow, mw2::CenteredMovingWindow)
    return nwindows(mw1) == nwindows(mw2)
end

# TODO: move this in SoleBase!!!
function _centered_moving_window(l::Integer, nw::Integer)
    bound_dist = l / (2*nw)
    # TODO: optimize!!!
    return [max(1, 1+round(Int, i*bound_dist)):min(l, l - round(Int, i*bound_dist)) for i in 0:(nw-1)]
end

function getwindows(v::AbstractArray, mw::CenteredMovingWindow)
    indices = zip(_centered_moving_window.(size(v), nwindows(mw))...)
    return [v[idxs...] for idxs in indices]
end
# TODO: tests for CenteredMovingWindow
