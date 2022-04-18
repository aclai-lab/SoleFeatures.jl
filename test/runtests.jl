using SoleBase
using SoleFeatures
using Test

include("./test_function.jl")

@testset "SoleFeatures.jl" begin

    @testset "Testing VarianceThreshold on temporal series in MultiFrameDataset" begin
        # test VarianceThreshold on temporal series in MultiFrameDataset
        # Variance threshold: 0.09
        # Expected behavior: Attributes "firstcol" and "fourthcol" should be removed

        df = fake_temporal_series_dataset()
        df_expected = df[:, [2,3]]
        mfd = SoleBase.MultiFrameDataset(df)
        mfd_expected = SoleBase.MultiFrameDataset(df_expected)

        vt = VarianceThreshold(0.09)
        mfd_generated = SoleFeatures.apply(mfd, vt)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
    end

end
