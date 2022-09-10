using SoleFeatures
using Test

include("./test_function.jl")

@testset "SoleFeatures.jl" begin

    df = fake_temporal_series_dataset()
    nf = SoleFeatures.minmax_normalize_wrapper(0.0, 1.0)

    @testset "Testing VarianceThreshold on temporal series in MultiFrameDataset" begin
        # test VarianceThreshold on temporal series in MultiFrameDataset
        # Variance threshold: 0.09
        # Expected behavior: Attributes "firstcol", "fourthcol" and "fifthcol" will be removed

        mfd = SoleBase.MultiFrameDataset(df)
        df_expected = df[:, [2,3]]
        mfd_expected = SoleBase.MultiFrameDataset(df_expected)

        vt = VarianceThreshold(0.09)
        mfd_generated = SoleFeatures.apply(mfd, vt; normalize_function = nf)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
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
        mfd_generated = SoleFeatures.apply(mfd, vt, frame_num; normalize_function = nf)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
    end

    @testset "Testing VarianceRanking on temporal series of frame of MultiFrameDataset" begin
        # test VarianceThreshold on temporal in MultiFrameDataset
        # Top selected features: 2
        # Expected behavior: Attributes "firstcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset(df)
        df_expected = df[:, [2,3]]
        mfd_expected = SoleBase.MultiFrameDataset(df_expected)

        vr = VarianceRanking(2)
        mfd_generated = SoleFeatures.apply(mfd, vr; normalize_function = nf)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
    end

    @testset "Testing VarianceRanking on temporal series of frame of MultiFrameDataset" begin
        # test VarianceRanking on temporal series of frame of MultiFrameDataset
        # Top selected features: 2
        # Expected behavior: Attributes "firstcol", "secondcol" and "fourthcol" will be removed

        mfd = SoleBase.MultiFrameDataset([[1,2,3,4],[5]], df)
        df_expected = df[:, [2,5]]
        mfd_expected = SoleBase.MultiFrameDataset([[1],[2]], df_expected)

        vr = VarianceRanking(1)
        frame_num = 1
        mfd_generated = SoleFeatures.apply(mfd, vr, frame_num; normalize_function = nf)

        @test SoleBase.isequal(mfd_expected, mfd_generated)
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
