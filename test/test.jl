include("/home/painkiller/develop/SoleFeatures.jl/src/SoleFeatures.jl")
using StatsBase

mtrx = rand(4,4)
cmtrx = cor(mtrx)
SoleFeatures.findcorrelation(cmtrx)
