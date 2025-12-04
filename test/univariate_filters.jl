using Test
using SoleFeatures

using Statistics
using DataTreatments
using Random
using SoleData: Artifacts

# fill your Artifacts.toml file;
# Artifacts.fillartifacts()
natopsloader = Artifacts.NatopsLoader()

Xts, yts = Artifacts.load(natopsloader)

# dataset with windowed features
dt = DataTreatment(Xts, :aggregate; 
                   win=(splitwindow(nwindows=3),),
                   features=(mean, std, maximum))

# normalize to have only positive values
X_grouped = grouped_norm(dt.dataset, DataTreatments.minmax(); 
                        featvec=get_vecfeatures(dt.featureid))

X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]

@testset "Chi2Filter Tests" begin
    @testset "chi2 function" begin
        chi2stats, pvalues = SoleFeatures.chi2(X, y)
        
        @test length(chi2stats) == size(X, 2)
        @test length(pvalues)   == size(X, 2)
        @test all(chi2stats .>= 0)     # chi-squared stats are non-negative
        @test all(0 .<= pvalues .<= 1) # p-values in [0, 1]
        
        @test chi2stats isa Vector{Float64}
        @test pvalues   isa Vector{Float64}
        
        # test error on negative values
        X_neg = copy(X)
        X_neg[1, 1] = -1
        @test_throws ArgumentError SoleFeatures.chi2(X_neg, y)
    end
    
    @testset "Chi2Filter construction" begin
        filter1 = Chi2Filter(IdentityLimiter())
        @test filter1 isa Chi2Filter
        @test filter1.limiter isa IdentityLimiter
        
        filter2 = Chi2Ranking(3)
        @test filter2 isa Chi2Filter
        @test filter2.limiter isa SoleFeatures.RankingLimiter
        
        filter3 = Chi2Threshold(alpha=0.05)
        @test filter3 isa Chi2Filter
        @test filter3.limiter isa SoleFeatures.ThresholdLimiter
        
        # test supervision properties
        @test SoleFeatures.is_supervised(filter1)
        @test !SoleFeatures.is_unsupervised(filter1)
    end
    
    @testset "Chi2Filter scoring" begin
        filter = Chi2Filter(IdentityLimiter())
        pvalues = SoleFeatures.score(filter, X_grouped, yts)
        
        @test length(pvalues) == size(X_grouped, 2)
        @test all(0 .<= pvalues .<= 1)
        @test pvalues isa Vector{Float64}
    end
end

@testset "FisherScoreFilter Tests" begin
    @testset "Fisher score function" begin
        indices, scores = SoleFeatures.fisher_score(X, y)
        
        @test length(indices) == size(X, 2)
        @test length(scores)  == size(X, 2)
        @test indices isa Vector{Int}
        @test scores  isa Vector{Float64}
        
        # Indices should be a permutation
        @test sort(indices) == 1:size(X, 2)
        
        # Scores should be sorted in ascending order
        @test issorted(scores[indices], rev=false)
    end
    
    @testset "FisherScoreFilter construction" begin
        filter1 = FisherScoreFilter(IdentityLimiter())
        @test filter1 isa FisherScoreFilter
        @test filter1.limiter isa IdentityLimiter
        
        filter2 = FisherScoreRanking(3)
        @test filter2 isa FisherScoreFilter
        @test filter2.limiter isa SoleFeatures.RankingLimiter
        
        filter3 = FisherScoreThreshold(alpha=0.05)
        @test filter3 isa FisherScoreFilter
        @test filter3.limiter isa SoleFeatures.ThresholdLimiter
        
        # test supervision properties
        @test  SoleFeatures.is_supervised(filter1)
        @test !SoleFeatures.is_unsupervised(filter1)
    end
    
    @testset "FisherScoreFilter scoring" begin
        filter = FisherScoreFilter(IdentityLimiter())
        scores = SoleFeatures.score(filter, dt.dataset, yts)
        
        @test length(scores) == size(dt.dataset, 2)
        @test scores isa Vector{Float64}
        @test all(isfinite.(scores))
    end
end

@testset "MutualInformationClassif Tests" begin
    @testset "Mutual information classifier function" begin
        scores = SoleFeatures.mutual_info_classifier(X, y)
        
        @test length(scores)  == size(X, 2)
        @test scores  isa Vector{Float64}
    end
    
    @testset "MutualInformationClassif construction" begin
        filter1 = MutualInformationClassif(IdentityLimiter())
        @test filter1 isa MutualInformationClassif
        @test filter1.limiter isa IdentityLimiter
        
        filter2 = MutualInformationClassifRanking(3)
        @test filter2 isa MutualInformationClassif
        @test filter2.limiter isa SoleFeatures.RankingLimiter
        
        filter3 = MutualInformationClassifThreshold(alpha=0.05)
        @test filter3 isa MutualInformationClassif
        @test filter3.limiter isa SoleFeatures.ThresholdLimiter
        
        # test supervision properties
        @test  SoleFeatures.is_supervised(filter1)
        @test !SoleFeatures.is_unsupervised(filter1)
    end
    
    @testset "MutualInformationClassif scoring" begin
        filter = MutualInformationClassif(IdentityLimiter())
        scores = SoleFeatures.score(filter, dt.dataset, yts; n_neighbors=2, rng=Random.Xoshiro(11))
        
        @test length(scores) == size(dt.dataset, 2)
        @test scores isa Vector{Float64}
        @test all(isfinite.(scores))
    end
end

@testset "PearsonCorFilter Tests" begin
    @testset "PearsonCorFilter construction" begin
        filter1 = get_pearson_cor_identity()
        @test filter1 isa PearsonCorFilter
        @test filter1.limiter isa IdentityLimiter

        filter2 = get_pearson_cor_threshold(0.5, >)
        @test filter2 isa PearsonCorFilter
        @test filter2.limiter isa SoleFeatures.ThresholdLimiter

        filter3 = get_pearson_cor_ranking(3,false)
        @test filter3 isa PearsonCorFilter
        @test filter3.limiter isa SoleFeatures.RankingLimiter

        filter4 = get_pearson_cor_ranking(3)
        @test filter4 isa PearsonCorFilter
        @test filter4.limiter isa SoleFeatures.RankingLimiter

        filter5 = get_pearson_cor_percentage(0.9,false)
        @test filter5 isa PearsonCorFilter
        @test filter5.limiter isa SoleFeatures.PercentageLimiter

        filter6 = get_pearson_cor_percentage(0.9)
        @test filter6 isa PearsonCorFilter
        @test filter6.limiter isa SoleFeatures.PercentageLimiter
        
        # test supervision properties
        for filter in (filter1, filter2, filter3, filter4, filter5, filter6)
            @test  SoleFeatures.is_supervised(filter)
            @test !SoleFeatures.is_unsupervised(filter)
        end
    end

    @testset "PearsonCorFilter scoring" begin
        filter = PearsonCorFilter(IdentityLimiter())
        scores = SoleFeatures.score(filter, Float64.(X), y)

        @test length(scores) == size(X, 2)
        @test scores isa Vector{Float64}
        @test all(isfinite.(scores))
    end
end

@testset "RandomFilter Tests" begin
    @testset "RandomFilter construction" begin
        filter1 = get_random_identity()
        @test filter1 isa RandomFilter
        @test filter1.limiter isa IdentityLimiter

        filter2 = get_random_identity(2)
        @test filter2 isa RandomFilter
        @test filter2.limiter isa IdentityLimiter

        filter3 = get_random_threshold(0.5, >)
        @test filter3 isa RandomFilter
        @test filter3.limiter isa SoleFeatures.ThresholdLimiter

        filter4 = get_random_threshold(0.5, >, 2)
        @test filter4 isa RandomFilter
        @test filter4.limiter isa SoleFeatures.ThresholdLimiter

        filter5 = get_random_ranking(3)
        @test filter5 isa RandomFilter
        @test filter5.limiter isa SoleFeatures.RankingLimiter
        
        filter6 = get_random_ranking(3, false)
        @test filter6 isa RandomFilter
        @test filter6.limiter isa SoleFeatures.RankingLimiter

        filter7 = get_random_ranking(3, false, 2)
        @test filter7 isa RandomFilter
        @test filter7.limiter isa SoleFeatures.RankingLimiter

        filter8 = get_random_percentage(0.9)
        @test filter8 isa RandomFilter
        @test filter8.limiter isa SoleFeatures.PercentageLimiter

        filter9 = get_random_percentage(0.9,false)
        @test filter9 isa RandomFilter
        @test filter9.limiter isa SoleFeatures.PercentageLimiter

        filter10 = get_random_percentage(0.9, false, 2)
        @test filter10 isa RandomFilter
        @test filter10.limiter isa SoleFeatures.PercentageLimiter
        
        # test supervision properties
        for filter in (filter1, filter2, filter3, filter4, filter5, filter6, 
            filter7, filter8, filter9, filter10
        )
            @test !SoleFeatures.is_supervised(filter)
            @test SoleFeatures.is_unsupervised(filter)
        end
    end

    @testset "RandomFilter scoring" begin
        filter = RandomFilter(IdentityLimiter(), nothing)
        scores = SoleFeatures.score(filter, Float64.(X))

        @test length(scores) == size(X, 2)
        @test scores isa Vector{Float64}
        @test all(isfinite.(scores))
    end
end