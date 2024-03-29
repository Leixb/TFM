using DrWatson, Test
@quickactivate "TFM"

# Here you include files using `srcdir`
# include(srcdir("resampling.jl"))

using TFM.Resampling

@testset "resampling tests" begin
    using MLJBase: train_test_pairs
    using MLJ

    @testset "TwoFold" begin
        cv = TwoFold(; rng=1234)
        (train1, test1), (train2, test2) = train_test_pairs(cv, 1:10)
        @test train1 == test2 == [6, 1, 10, 2, 3]
        @test train2 == test1 == [9, 5, 7, 4, 8]
    end

    @testset "RepeatedCV" begin
        cv = RepeatedCV{TwoFold}(10, 5678)
        result = train_test_pairs(cv, 1:10)
        @test result == [
            ([7, 2, 4, 10, 9], [6, 8, 5, 3, 1]), ([6, 8, 5, 3, 1], [7, 2, 4, 10, 9]),
            ([10, 1, 4, 3, 9], [7, 8, 6, 5, 2]), ([7, 8, 6, 5, 2], [10, 1, 4, 3, 9]),
            ([10, 2, 6, 8, 5], [1, 9, 7, 4, 3]), ([1, 9, 7, 4, 3], [10, 2, 6, 8, 5]),
            ([9, 7, 8, 4, 6], [5, 1, 3, 10, 2]), ([5, 1, 3, 10, 2], [9, 7, 8, 4, 6]),
            ([4, 7, 8, 3, 2], [5, 6, 1, 10, 9]), ([5, 6, 1, 10, 9], [4, 7, 8, 3, 2]),
            ([8, 9, 1, 2, 7], [10, 3, 6, 5, 4]), ([10, 3, 6, 5, 4], [8, 9, 1, 2, 7]),
            ([10, 8, 9, 2, 3], [1, 4, 7, 5, 6]), ([1, 4, 7, 5, 6], [10, 8, 9, 2, 3]),
            ([8, 6, 4, 10, 2], [7, 1, 5, 9, 3]), ([7, 1, 5, 9, 3], [8, 6, 4, 10, 2]),
            ([9, 10, 4, 8, 7], [2, 5, 3, 6, 1]), ([2, 5, 3, 6, 1], [9, 10, 4, 8, 7]),
            ([4, 8, 7, 9, 1], [3, 2, 6, 5, 10]), ([3, 2, 6, 5, 10], [4, 8, 7, 9, 1])
        ]

        cv = RepeatedCV{TwoFold}(10, 5678)
        result = train_test_pairs(cv, 1:10)
        @test result == [
            ([7, 2, 4, 10, 9], [6, 8, 5, 3, 1]), ([6, 8, 5, 3, 1], [7, 2, 4, 10, 9]),
            ([10, 1, 4, 3, 9], [7, 8, 6, 5, 2]), ([7, 8, 6, 5, 2], [10, 1, 4, 3, 9]),
            ([10, 2, 6, 8, 5], [1, 9, 7, 4, 3]), ([1, 9, 7, 4, 3], [10, 2, 6, 8, 5]),
            ([9, 7, 8, 4, 6], [5, 1, 3, 10, 2]), ([5, 1, 3, 10, 2], [9, 7, 8, 4, 6]),
            ([4, 7, 8, 3, 2], [5, 6, 1, 10, 9]), ([5, 6, 1, 10, 9], [4, 7, 8, 3, 2]),
            ([8, 9, 1, 2, 7], [10, 3, 6, 5, 4]), ([10, 3, 6, 5, 4], [8, 9, 1, 2, 7]),
            ([10, 8, 9, 2, 3], [1, 4, 7, 5, 6]), ([1, 4, 7, 5, 6], [10, 8, 9, 2, 3]),
            ([8, 6, 4, 10, 2], [7, 1, 5, 9, 3]), ([7, 1, 5, 9, 3], [8, 6, 4, 10, 2]),
            ([9, 10, 4, 8, 7], [2, 5, 3, 6, 1]), ([2, 5, 3, 6, 1], [9, 10, 4, 8, 7]),
            ([4, 8, 7, 9, 1], [3, 2, 6, 5, 10]), ([3, 2, 6, 5, 10], [4, 8, 7, 9, 1])
        ]


        cv = RepeatedCV{CV}(3, 9124)
        result = train_test_pairs(cv, 1:20)
        @test result == [
            ([5, 16, 12, 8, 19, 2, 17, 20, 3, 15, 11, 13, 4, 1, 14, 7], [6, 10, 18, 9]),
            ([6, 10, 18, 9, 19, 2, 17, 20, 3, 15, 11, 13, 4, 1, 14, 7], [5, 16, 12, 8]),
            ([6, 10, 18, 9, 5, 16, 12, 8, 20, 3, 15, 11, 13, 4, 1, 14, 7], [19, 2, 17]),
            ([6, 10, 18, 9, 5, 16, 12, 8, 19, 2, 17, 11, 13, 4, 1, 14, 7], [20, 3, 15]),
            ([6, 10, 18, 9, 5, 16, 12, 8, 19, 2, 17, 20, 3, 15, 1, 14, 7], [11, 13, 4]),
            ([6, 10, 18, 9, 5, 16, 12, 8, 19, 2, 17, 20, 3, 15, 11, 13, 4], [1, 14, 7]),
            ([20, 9, 13, 12, 4, 6, 15, 7, 16, 19, 14, 2, 1, 11, 17, 5], [3, 18, 8, 10]),
            ([3, 18, 8, 10, 4, 6, 15, 7, 16, 19, 14, 2, 1, 11, 17, 5], [20, 9, 13, 12]),
            ([3, 18, 8, 10, 20, 9, 13, 12, 7, 16, 19, 14, 2, 1, 11, 17, 5], [4, 6, 15]),
            ([3, 18, 8, 10, 20, 9, 13, 12, 4, 6, 15, 14, 2, 1, 11, 17, 5], [7, 16, 19]),
            ([3, 18, 8, 10, 20, 9, 13, 12, 4, 6, 15, 7, 16, 19, 11, 17, 5], [14, 2, 1]),
            ([3, 18, 8, 10, 20, 9, 13, 12, 4, 6, 15, 7, 16, 19, 14, 2, 1], [11, 17, 5]),
            ([9, 5, 7, 19, 11, 8, 17, 4, 12, 14, 6, 20, 18, 13, 10, 1], [3, 15, 16, 2]),
            ([3, 15, 16, 2, 11, 8, 17, 4, 12, 14, 6, 20, 18, 13, 10, 1], [9, 5, 7, 19]),
            ([3, 15, 16, 2, 9, 5, 7, 19, 4, 12, 14, 6, 20, 18, 13, 10, 1], [11, 8, 17]),
            ([3, 15, 16, 2, 9, 5, 7, 19, 11, 8, 17, 6, 20, 18, 13, 10, 1], [4, 12, 14]),
            ([3, 15, 16, 2, 9, 5, 7, 19, 11, 8, 17, 4, 12, 14, 13, 10, 1], [6, 20, 18]),
            ([3, 15, 16, 2, 9, 5, 7, 19, 11, 8, 17, 4, 12, 14, 6, 20, 18], [13, 10, 1])
        ]
    end

    @testset "FiveTwo" begin
        cv = FiveTwo(89767)
        result = train_test_pairs(cv, 1:20)
        @test result == [
            ([2, 8, 19, 3, 1, 15, 4, 11, 18, 17], [16, 12, 10, 7, 14, 13, 6, 5, 9, 20]),
            ([16, 12, 10, 7, 14, 13, 6, 5, 9, 20], [2, 8, 19, 3, 1, 15, 4, 11, 18, 17]),
            ([13, 15, 8, 10, 4, 17, 1, 5, 14, 2], [11, 16, 19, 12, 7, 20, 9, 18, 3, 6]),
            ([11, 16, 19, 12, 7, 20, 9, 18, 3, 6], [13, 15, 8, 10, 4, 17, 1, 5, 14, 2]),
            ([6, 20, 16, 13, 18, 4, 14, 11, 8, 2], [15, 1, 5, 17, 12, 19, 9, 3, 7, 10]),
            ([15, 1, 5, 17, 12, 19, 9, 3, 7, 10], [6, 20, 16, 13, 18, 4, 14, 11, 8, 2]),
            ([1, 8, 17, 14, 6, 13, 16, 9, 15, 2], [4, 18, 20, 12, 11, 5, 3, 19, 10, 7]),
            ([4, 18, 20, 12, 11, 5, 3, 19, 10, 7], [1, 8, 17, 14, 6, 13, 16, 9, 15, 2]),
            ([1, 11, 3, 4, 16, 6, 20, 2, 15, 12], [14, 9, 10, 19, 18, 5, 13, 7, 8, 17]),
            ([14, 9, 10, 19, 18, 5, 13, 7, 8, 17], [1, 11, 3, 4, 16, 6, 20, 2, 15, 12])
        ]
    end
end

@testset "transformers" begin
    @testset "top n transformer" begin
        using MLJ
        using TFM.Transformers
        X = (vendor=categorical(["IBM", "HP", "HP", "Asus", "IBM", "honeywell", "hello", "IBM"]),
            height=[1.85, 1.67, 1.5, 1.67, 1.85, 1.67, 1.5, 1.67],
            grade=categorical(["A", "B", "A", "B", "A", "B", "B", "A"], ordered=true),
            n_devices=[3, 2, 4, 3, 3, 2, 4, 3])
        trans = TopCatTransformer(n=3)
        mach = machine(trans, X)
        @test_logs (:info, "Training machine(TopCatTransformer(features = Symbol[], …), …).") fit!(mach)

        W = transform(mach, X)

        @test W.vendor == categorical(["IBM", "HP", "HP", "Asus", "IBM", "OTHER", "OTHER", "IBM"])
        @test W.grade == X.grade
        @test W.height == X.height
        @test W.n_devices == X.n_devices

        Xtest = (
            vendor=categorical(["IBM", "HP", "DD", "DD", "IBM", "honeywell", "hello", "IBM"]),
            height=[1.85, 1.67, 1.5, 1.67, 1.85, 1.7, 1.5, 1.67],
            grade=categorical(["A", "B", "A", "D", "A", "B", "B", "A"], ordered=true),
            n_devices=[3, 2, 4, 3, 3, 2, 4, 3]
        )
        Wtest = transform(mach, Xtest)

        @test Wtest.vendor == categorical(["IBM", "HP", "OTHER", "OTHER", "IBM", "OTHER", "OTHER", "IBM"])
        @test Wtest.grade == categorical(["A", "B", "A", "OTHER", "A", "B", "B", "A"], ordered=true)

        trans = TopCatTransformer(n=2, other="Other")
        mach = machine(trans, X)
        @test_logs (:info, "Training machine(TopCatTransformer(features = Symbol[], …), …).") fit!(mach)

        W = transform(mach, X)
        @test W.vendor == categorical(["IBM", "HP", "HP", "Other", "IBM", "Other", "Other", "IBM"])
        @test W.grade == X.grade
        @test W.height == X.height
        @test W.n_devices == X.n_devices
    end

    @testset "multiplier transformer" begin
        using MLJ
        using TFM.Transformers

        X = [
            1 2 3 4 5 6
            2 3 4 5 6 7
            3 4 5 6 7 8
            4 5 6 7 8 9
        ]

        trans = Multiplier(factor=2.5)
        mach = machine(trans)
        W = transform(mach, X)

        @test W == [
            2.5 5.0 7.5 10.0 12.5 15.0
            5.0 7.5 10.0 12.5 15.0 17.5
            7.5 10.0 12.5 15.0 17.5 20.0
            10.0 12.5 15.0 17.5 20.0 22.5
        ]
    end
end

@testset "datasets" begin
    using TFM.DataSets
    using TFM.Models
    map(DataSets.all) do ds

        # Check that the target is not dropped
        @test DataSets.select_columns(ds) === nothing || DataSets.target(ds) in DataSets.select_columns(ds)
        @test DataSets.drop_columns(ds) === nothing || !(DataSets.target(ds) in DataSets.drop_columns(ds))

        # All datasets can be loaded, partitioned and the machines built
        # without warnings
        @test_nowarn begin
            pipe = Models.pipeline(ds)
            (Xtrain, Xtest), (ytrain, ytest) = partition(ds)
            mach = machine(pipe, Xtrain, ytrain)
        end
    end
end
