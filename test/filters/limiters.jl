using Test
using SoleFeatures

X = [1 1 3; 0 1 5; 5 4 1; 6 6 2; 1 4 0; 0 0 0]
y = [1, 1, 0, 0, 2, 2]
chi2stats = Chi2Filter(X, y)

@testset "ThresholdLimiter" begin
    # Basic test: >
    tl1 = ThresholdLimiter(chi2stats; ordf=(>), threshold=7)
    @test isa(tl1, ThresholdLimiter)
    @test all(get_score(chi2stats)[tl1.rank] .> 7)

    # Test: <
    tl2 = ThresholdLimiter(chi2stats; ordf=(<), threshold=7)
    @test all(get_score(chi2stats)[tl2.rank] .< 7)

    # Test: ≥
    tl3 = ThresholdLimiter(chi2stats; ordf=(≥), threshold=7)
    @test all(get_score(chi2stats)[tl3.rank] .≥ 7)

    # Test: ≤
    tl4 = ThresholdLimiter(chi2stats; ordf=(≤), threshold=7)
    @test all(get_score(chi2stats)[tl4.rank] .≤ 7)

    # Error: invalid operator
    @test_throws DomainError ThresholdLimiter(chi2stats; ordf=(==), threshold=7)
end

@testset "RankingLimiter" begin
    # Default: top 2, descending
    rl1 = RankingLimiter(chi2stats; nbest=2)
    scores = get_score(chi2stats)
    sorted_scores = sort(scores; rev=true)
    @test scores[rl1.rank] == sorted_scores[1:2]

    # Ascending order
    rl2 = RankingLimiter(chi2stats; nbest=2, rev=false)
    sorted_scores_asc = sort(scores; rev=false)
    @test scores[rl2.rank] == sorted_scores_asc[1:2]

    # Error: nbest <= 0
    @test_throws DomainError RankingLimiter(chi2stats; nbest=0)
    @test_throws DomainError RankingLimiter(chi2stats; nbest=-1)
end