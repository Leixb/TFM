using MLJ
using Distributions

function paired_ttest_5x2cv(perfA::Vector{Float64}, perfB::Vector{Float64})
    """
    Perform a paired t-test on the results of two estimators.

    perfA: Vector of performance scores for estimator A
    perfB: Vector of performance scores for estimator B

    Both vectors must have length 10, and be derived from the
    same 5x2 cross validation. Where perfA[1] is the performance of
    the train set of fold 1 of estimator A, and perfA[2] is the
    performance of the test set of fold 1 of estimator A, and so on.

    References:
        - https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
        - Dietterich TG (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Comput 10:1895–1923.
    """

    @assert length(perfA) == 10
    @assert length(perfB) == 10

    diff = perfA .- perfB

    diffTrain = diff[1:2:end]
    diffTest = diff[2:2:end]

    meanP = (diffTrain .+ diffTest) ./ 2

    variance = (diffTrain .- meanP) .^ 2 + (diffTest .- meanP) .^ 2

    t = diffTrain[1] / sqrt(sum(variance) / 5)

    pvalue = 2 * (1 - cdf(TDist(5), abs(t)))

    return t, pvalue
end
