using SoleFeatures, SoleXplorer
using Test
using Sole
using Random, StatsBase, DataFrames
using MLJ

# ---------------------------------------------------------------------------- #
#                             DATASET PREPARATION                              #
# ---------------------------------------------------------------------------- #
train_seed = 11

X, y       = SoleData.load_arff_dataset("NATOPS")

features = complete_set
type = adaptivewindow
nwindows = 6
relative_overlap = 0.2

Xdf, Xinfo = @test_nowarn SoleFeatures.feature_selection_preprocess(X; features, type, nwindows, relative_overlap)

fs = @test_nowarn feature_selection(Xdf, y, Xinfo)

aggrby = (
    aggrby = (:feat,),
    aggregatef = mean,
    group_before_score = true,
)
fs_methods = [
    ( # STEP 1: unsupervised variance-based filter
        selector = PearsonCorFilter(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # STEP 2: supervised Mutual Information filter
        selector = MutualInformationClassif(IdentityLimiter()),
        limiter = PercentageLimiter(0.75),
    ),
    ( # STEP 3: group results by variable
        selector = IdentityFilter(),
        limiter = IdentityLimiter(),
    ),
]
norm::Bool = false

fs = @test_nowarn feature_selection(Xdf, y, Xinfo; aggrby, fs_methods, norm)

rng = Random.Xoshiro(train_seed)
Random.seed!(rng, train_seed)
nofs_model = @test_nowarn traintest(X, y; models=(; type=:decisiontree, rng=rng))
nofs_preds = MLJ.predict(nofs_model.mach, nofs_model.ds.Xtest)
nofs_yhat = MLJ.mode.(nofs_preds)
nofs_acc = MLJ.accuracy(nofs_yhat, nofs_model.ds.ytest)

rng = Random.Xoshiro(train_seed)
idxes = [f.id for f in fs[2]]
Random.seed!(rng, train_seed)
fs_model = @test_nowarn traintest(Xdf[:, idxes], y; models=(; type=:decisiontree, rng=rng))
fs_preds = MLJ.predict(fs_model.mach, fs_model.ds.Xtest)
fs_yhat = MLJ.mode.(fs_preds)
fs_acc = MLJ.accuracy(fs_yhat, fs_model.ds.ytest)

@show size(Xdf, 2)
@show size(Xdf[:, idxes], 2)
@show nofs_acc fs_acc
# @test nofs_acc < fs_acc
