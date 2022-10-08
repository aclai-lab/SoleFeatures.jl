using SoleFeatures
using BenchmarkTools
using StatsBase

include("./test_function.jl")

frame_num = 1
ninstances = 10000
nattr = 300
ts_len = 100

println("Start time test")

# println("Random generation...")
# @benchmarkable mfd = random_timeseries_mfd(;ninstances=10000, nattr=600, ts_len=100);

# println("Min max apply...")
# @benchmarkable  nmfd = SoleFeatures.minmax_normalize(mfd, 1);

# vt = VarianceThreshold(0.09)

# println("VarianceThreshold build bitmask...")
# bm, fr_bm = @benchmarkable  SoleFeatures.buildbitmask(nmfd, frame_num, vt) setup=(mfd = random_timeseries_mfd(;ninstances=10000, nattr=300, ts_len=100);)

# println("VarianceThreshold apply bitmask...")
# @benchmarkable  mfd_generated = SoleFeatures.transform(mfd, bm)

# nattr = SoleBase.SoleDataset.nattributes(mfd, 1)

# test on

# n = Integer(ceil(nattr * 0.75))
# cr = CorrelationRanking(n, :pearson)

# mfd = random_timeseries_mfd(;ninstances=ninstances, nattr=nattr, ts_len=ts_len)
# bm = build_fake_bit_mask(nattr)

# println("CorrelationRanking build bitmask...")
# b = @benchmark SoleFeatures.buildbitmask(mfd, frame_num, cr) setup=(mfd=$mfd)
# show(stdout, MIME"text/plain"(), b)
# println()

# println("CorrelationRanking apply bitmask...")
# b = @benchmark SoleFeatures.transform(mfd, bm) setup=(mfd=$mfd, bm=$bm)
# show(stdout, MIME"text/plain"(), b)
# println()

df = random_timeseries_df(;ninstances=ninstances, nattr=nattr, ts_len=ts_len)
io = open("/home/painkiller/develop/SoleFeatures.jl/test/output.info", "w")

println(io, "Start time experimenti:")
println(io, "")
println(io, "Correlation without memory_saving")
flush(io)
b = @benchmark SoleFeatures.correlation($df, StatsBase.cor; memory_saving=false) samples=2
show(io, MIME"text/plain"(), b)
flush(io)

println(io, "")
println(io, "Correlation with memory_saving")
flush(io)
b = @benchmark SoleFeatures.correlation($df, StatsBase.cor; memory_saving=true) samples=2
show(io, MIME"text/plain"(), b)
close(io)
