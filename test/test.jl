using SoleFeatures

include("./test_function.jl")

df = fake_temporal_series_dataset()
ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)

mfd = SoleBase.MultiFrameDataset(ndf)
dfe = ndf[:, [2,3]]
mfde = SoleBase.MultiFrameDataset(dfe)

vr = VarianceRanking(2)
bm = buildbitmask(mfd, )
mfdg = SoleFeatures.transform(mfd, vr)

dfg = SoleBase.SoleDataset.data(mfdg)
