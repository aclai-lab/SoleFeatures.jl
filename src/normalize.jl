# ---------------------------------------------------------------------------- #
#                      @_Normalization extended methods                        #
# ---------------------------------------------------------------------------- #
scale(s) = Base.Fix2(/, s)
@_Normalization Scale (std,) scale
@_Normalization ScaleMad ((x)->mad(x; normalize=false),) scale
@_Normalization ScaleFirst (first,) scale

@_Normalization PNorm1 ((x)->norm(x, 1),) scale
@_Normalization PNorm ((x)->norm(x, 2),) scale
@_Normalization PNormInf ((x)->norm(x, Inf),) scale
