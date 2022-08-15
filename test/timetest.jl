using SoleFeatures
using BenchmarkTools

include("./test_function.jl")

frame_num = 1
ninstances = 10000
nattr = 600
ts_len = 100

println("Start time test")

# println("Random generation...")
# @benchmarkable mfd = random_timeseries_mfd(;ninstances=10000, nattr=600, ts_len=100);

# println("Min max apply...")
# @benchmarkable  nmfd = SoleFeatures.minmax_normalize(mfd, 1);

# vt = VarianceThreshold(0.09)

# println("VarianceThreshold build bitmask...")
# bm, fr_bm = @benchmarkable  SoleFeatures.build_bitmask(nmfd, frame_num, vt) setup=(mfd = random_timeseries_mfd(;ninstances=10000, nattr=300, ts_len=100);)

# println("VarianceThreshold apply bitmask...")
# @benchmarkable  mfd_generated = SoleFeatures.apply(mfd, bm)

# nattr = SoleBase.SoleDataset.nattributes(mfd, 1)

n = Integer(ceil(nattr * 0.75))
cr = CorrelationRanking(n, :pearson)

mfd = random_timeseries_mfd(;ninstances=ninstances, nattr=nattr, ts_len=ts_len)
bm = build_fake_bit_mask(nattr)

println("CorrelationRanking build bitmask...")
b = @benchmark SoleFeatures.build_bitmask(mfd, frame_num, cr) setup=(mfd=$mfd)
show(stdout, MIME"text/plain"(), b)
println()

println("CorrelationRanking apply bitmask...")
b = @benchmark SoleFeatures.apply(mfd, bm) setup=(mfd=$mfd, bm=$bm)
show(stdout, MIME"text/plain"(), b)
println()
