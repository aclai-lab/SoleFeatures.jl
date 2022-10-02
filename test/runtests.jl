using SoleFeatures
using Test

include("./test_function.jl")

@testset "SoleFeatures.jl" begin

    df = fake_temporal_series_dataset()
    ndf = SoleFeatures.minmax_normalize(df; min_quantile=0.0, max_quantile=1.0)
    # nf = SoleFeatures.minmax_normalize_wrapper(0.0, 1.0)

    @testset "Testing VarianceThreshold on temporal series in MultiFrameDataset" begin
        # test VarianceThreshold on temporal series in MultiFrameDataset
        # Variance threshold: 0.09
        # Expected behavior: Attributes "firstcol", "fourthcol" and "fifthcol" will be removed

        mfd = SoleBase.MultiFrameDataset(ndf)
        dfe = ndf[:, [2,3]]
        mfde = SoleBase.MultiFrameDataset(dfe)

        vt = VarianceThreshold(0.09)
        mfdg = SoleFeatures.apply(mfd, vt)
        dfg = SoleBase.SoleDataset.data(mfdg)

        @test isequal(dfe, dfg)
    end

    @testset "Testing VarianceThreshold on temporal series of frame of MultiFrameDataset" begin
        # test VarianceThreshold on temporal series of frame of MultiFrameDataset
        # Variance threshold: 0.09
        # Expected behavior: Attributes "firstcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset([[1,2,3,4],[5]], ndf)
        dfe = ndf[:, [2,3,5]]
        mfde = SoleBase.MultiFrameDataset([[1,2],[3]], dfe)

        vt = VarianceThreshold(0.09)
        mfdg = SoleFeatures.apply(mfd, vt, 1)
        dfg = SoleBase.SoleDataset.data(mfdg)

        @test isequal(dfe, dfg)
    end

    @testset "Testing VarianceRanking on temporal series of frame of MultiFrameDataset" begin
        # test VarianceThreshold on temporal in MultiFrameDataset
        # Top selected features: 2
        # Expected behavior: Attributes "firstcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset(ndf)
        dfe = ndf[:, [2,3]]
        mfde = SoleBase.MultiFrameDataset(dfe)

        vr = VarianceRanking(2)
        mfdg = SoleFeatures.apply(mfd, vr)
        dfg = SoleBase.SoleDataset.data(mfdg)

        @test isequal(dfe, dfg)
    end

    @testset "Testing VarianceRanking on temporal series of frame of MultiFrameDataset" begin
        # test VarianceRanking on temporal series of frame of MultiFrameDataset
        # Top selected features: 2
        # Expected behavior: Attributes "firstcol", "secondcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset([[1,2,3,4],[5]], ndf)
        dfe = ndf[:, [2,5]]
        mfde = SoleBase.MultiFrameDataset([[1],[2]], dfe)

        vr = VarianceRanking(1)
        mfdg = SoleFeatures.apply(mfd, vr, 1)
        dfg = SoleBase.SoleDataset.data(mfdg)

        @test isequal(dfe, dfg)
    end

    @testset "Testing VarianceThreshold on temporal series of frame of MultiFrameDataset" begin
        # test VarianceThreshold on temporal series of frame of MultiFrameDataset
        # Variance threshold: 0.09
        # Expected behavior: Attributes "firstcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset([[1,2,3,4],[5]], df)
        df_expected = df[:, [2,3,5]]
        mfd_expected = SoleBase.MultiFrameDataset([[1,2],[3]], df_expected)

        vt = VarianceThreshold(0.09)
        frame_num = 1

        nmfd = SoleFeatures.minmax_normalize(mfd, frame_num)
        bm, fr_bm = SoleFeatures.build_bitmask(nmfd, frame_num, vt)
        mfd_generated = SoleFeatures.apply(mfd, bm)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
    end



end
