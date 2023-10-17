#!/usr/bin/env julia

using TFM, TFM.Plots
using TFM.Experiments: experiment_data

using DataFramesMeta
using DrWatson

using CairoMakie

start_time = time()

@info "Activating CairoMakie with LaTeX theme"
CairoMakie.activate!()
Plots.tex_theme!()
Plots.no_color_cycle!()

opts_big = (;
    resolution=(1400, 800),
    dims=(8, 4)
)

opts_big_vert = (;
    resolution=(800, 1100),
    dims=(4, 8)
)
show_bands = true
if !@isdefined scan_dirs
    scan_dirs = "--scan" in ARGS || "-s" in ARGS
end
# WARN: --scan only affects Experiment 3, since 1 and 2 have already been
# purged.

if !@isdefined plot_list
    plot_list = 1:10
else
    @info "Plot list: $plot_list"
end

@info scan_dirs ? "Scanning directories" : "Not scanning directories (pass --scan to scan for new data)"

1 in plot_list && let
    @info "Benchmarks"
    @saveplot benchmark_time_inst = Plots.plot_benchmark_time_instances()
    @saveplot benchmark_time_improvement_old = Plots.exec_improvement()
end

2 in plot_list && let
    @info "Experiment run 1 (svms/) MSE (Frenay grid)"
    local opts = (;
        linkxaxes=true, linkyaxes=false,
        show_rbf=true
    )
    local mse = @chain experiment_data("svms", false) begin
        @rsubset(:dataset isa DataSets.DataSet)
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma])
        @rename(:measure_test = :measurement)
        Plots.regression
    end
    @saveplot MSE_all = Plots.plot_sigma(mse; opts..., show_bands)

    @saveplot MSE_frenay = Plots.plot_sigma(@rsubset(mse, :dataset isa DataSets.Frenay), ["AsinNorm"]; opts..., show_bands)
    @saveplot MSE_frenay_original = Plots.plot_frenay()
end

# measure_test is the result of the evaluation on the test set
# measurement is the result of the evaluation when doing cross-validation
# std is the standard deviation of the measurement

3 in plot_list && let
    @info "Experiment run 2 (svms_2/) Normalized RMSE"
    local opts = (;
        linkxaxes=true, linkyaxes=false,
        show_rbf=true
    )
    local nrmse = @chain experiment_data("svms_2", false) begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma], by=:measurement)
        Plots.regression()
        @rtransform(:n_feat = Plots.num_features[:dataset_cat])
        @transform(:corrected_sigma = :sigma .* :n_feat)
    end
    @saveplot nRMSE_all = Plots.plot_sigma(nrmse; opts..., opts_big_vert..., vertical=true)

    @saveplot nRMSE_asin = Plots.plot_sigma(nrmse, ["Asin"]; opts..., opts_big_vert..., vertical=true)
    @saveplot nRMSE_asinnorm = Plots.plot_sigma(nrmse, ["AsinNorm"]; opts..., opts_big_vert..., vertical=true)

    @saveplot nRMSE_frenay = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Frenay); opts...)
    @saveplot nRMSE_frenay_s = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Small); opts...)
    @saveplot nRMSE_frenay_l = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Large); opts...)
    @saveplot nRMSE_bank = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Bank); opts...)
    @saveplot nRMSE_pumadyn = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Pumadyn); opts...)
end

4 in plot_list && let
    @info "Experiment run 3 (svms3/) nRMSE with scaled sigma"
    local data_ex3 = experiment_data("svms3", scan_dirs)

    local nrmse_s = @chain data_ex3 begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma], by=:measure_test)
        Plots.regression()
        @rtransform(:n_feat = Plots.num_features[:dataset_cat])
        @transform(:corrected_sigma = :sigma .* :n_feat)
    end

    local nrmse_s_nodelve = @rsubset(nrmse_s, !(:dataset isa DataSets.Delve))

    local opts = Dict(
        :linkyaxes => true,
        :linkxaxes => true,
        :sigma => :corrected_sigma,
        :measure => :measure_cv,
        :std => :measure_std,
        :show_bands => true,
        :show_rbf => true
    )

    @info "Experiment run 3 (smvs3/) - part 1: Regression plots"

    @saveplot nRMSE_all_scaled = Plots.plot_sigma(nrmse_s; opts..., show_bands, opts_big_vert..., vertical=true)
    #
    # @saveplot nRMSE_asin_scaled = Plots.plot_sigma(nrmse_s, ["Asin"]; opts..., show_bands, opts_big_vert..., vertical=true)
    # @saveplot nRMSE_asinnorm_scaled = Plots.plot_sigma(nrmse_s, ["AsinNorm"]; opts..., show_bands, opts_big_vert..., vertical=true)

    @saveplot nRMSE_nodelve_all_scaled = Plots.plot_sigma(nrmse_s_nodelve; opts..., show_bands, opts_big.resolution)

    @saveplot nRMSE_nodelve_asin_scaled = Plots.plot_sigma(nrmse_s_nodelve, ["Asin"]; opts..., show_bands, opts_big.resolution)
    @saveplot nRMSE_nodelve_asinnorm_scaled = Plots.plot_sigma(nrmse_s_nodelve, ["AsinNorm"]; opts..., show_bands, opts_big.resolution)

    # @saveplot nRMSE_frenay_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay); opts..., show_bands)
    # @saveplot nRMSE_frenay_s_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Small); opts..., show_bands)
    # @saveplot nRMSE_frenay_l_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Large); opts..., show_bands)

    # opts[:linkyaxes] = true

    # @saveplot nRMSE_delve_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32; opts...)
    # @saveplot nRMSE_delve_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8; opts...)
    # @saveplot nRMSE_delve_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32; opts...)
    # @saveplot nRMSE_delve_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8; opts...)
    #
    # @saveplot nRMSE_delve_asin_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32, ["Asin"]; opts...)
    # @saveplot nRMSE_delve_asin_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8, ["Asin"]; opts...)
    # @saveplot nRMSE_delve_asin_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32, ["Asin"]; opts...)
    # @saveplot nRMSE_delve_asin_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8, ["Asin"]; opts...)
    #
    # @saveplot nRMSE_delve_asinnorm_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32, ["AsinNorm"]; opts...)
    # @saveplot nRMSE_delve_asinnorm_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8, ["AsinNorm"]; opts...)
    # @saveplot nRMSE_delve_asinnorm_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32, ["AsinNorm"]; opts...)
    # @saveplot nRMSE_delve_asinnorm_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8, ["AsinNorm"]; opts...)

    @saveplot nRMSE_delve_all_scaled = Plots.plot_delve_all(nrmse_s; opts...)
    @saveplot nRMSE_delve_asinnorm_scaled = Plots.plot_delve_all(nrmse_s, ["AsinNorm"]; opts...)
    @saveplot nRMSE_delve_asin_scaled = Plots.plot_delve_all(nrmse_s, ["Asin"]; opts...)

    opts[:linkyaxes] = false

    @saveplot nRMSE_nodelve_all_scaled_unlinked = Plots.plot_sigma(nrmse_s_nodelve; opts..., show_bands, opts_big.resolution, vertical=true)

    @saveplot nRMSE_nodelve_asin_scaled_unlinked = Plots.plot_sigma(nrmse_s_nodelve, ["Asin"]; opts..., show_bands, opts_big.resolution)
    @saveplot nRMSE_nodelve_asinnorm_scaled_unlinked = Plots.plot_sigma(nrmse_s_nodelve, ["AsinNorm"]; opts..., show_bands, opts_big.resolution)

    @saveplot nRMSE_delve_all_scaled_unlinked = Plots.plot_delve_all(nrmse_s; opts...)
    @saveplot nRMSE_delve_asinnorm_scaled_unlinked = Plots.plot_delve_all(nrmse_s, ["AsinNorm"]; opts...)
    @saveplot nRMSE_delve_asin_scaled_unlinked = Plots.plot_delve_all(nrmse_s, ["Asin"]; opts...)

    # opts[:linkyaxes] = true

    # local kernels = ["Acos0", "Acos1", "Acos2"]
    local kernels = ["AsinNorm", "Acos1", "Acos2"]

    # @saveplot nRMSE_nodelve_acos_scaled = Plots.plot_sigma(nrmse_s_nodelve, kernels; opts..., show_bands, opts_big.resolution, vertical=true)
    @saveplot nRMSE_nodelve_acos_scaled = Plots.plot_sigma(nrmse_s_nodelve, kernels; opts..., show_bands, opts_big.resolution, show_horizontal=["Acos1Norm", "Acos2Norm"])

    @saveplot nRMSE_nodelve_acos1_scaled = Plots.plot_sigma(nrmse_s_nodelve, ["Acos1"]; opts..., show_bands, opts_big.resolution, show_horizontal=["Acos1Norm"])
    @saveplot nRMSE_nodelve_acos2_scaled = Plots.plot_sigma(nrmse_s_nodelve, ["Acos2"]; opts..., show_bands, opts_big.resolution, show_horizontal=["Acos2Norm"])

    @saveplot nRMSE_acos1_scaled = Plots.plot_sigma(nrmse_s, ["Acos1"]; opts..., show_bands, opts_big_vert..., show_horizontal=["Acos1Norm"])
    @saveplot nRMSE_acos2_scaled = Plots.plot_sigma(nrmse_s, ["Acos2"]; opts..., show_bands, opts_big.resolution, show_horizontal=["Acos2Norm"])

    @saveplot nRMSE_acos_all_scaled = Plots.plot_sigma(nrmse_s, kernels; opts..., opts_big_vert...)
    # @saveplot nRMSE_acos_frenay_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay), kernels; opts...)
    # @saveplot nRMSE_acos_frenay_s_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Small), kernels; opts...)
    # @saveplot nRMSE_acos_frenay_l_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Large), kernels; opts...)

    # @saveplot nRMSE_acos_delve_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32, kernels; opts...)
    # @saveplot nRMSE_acos_delve_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8, kernels; opts...)
    # @saveplot nRMSE_acos_delve_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32, kernels; opts...)
    # @saveplot nRMSE_acos_delve_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8, kernels; opts...)

    @saveplot nRMSE_acos_delve = Plots.plot_delve_all(nrmse_s, kernels; opts...)

    @info "Experiment run 3 (smvs3/) - part 2: Classification plots"

    local acc_s = @chain data_ex3 begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma]; by=:measurement, maximum=true)
        Plots.classification()
        @rtransform(:n_feat = Plots.num_features[:dataset_cat])
        @transform(:corrected_sigma = :sigma .* :n_feat)
    end

    @saveplot accuracy_class_all_scaled = Plots.plot_sigma(acc_s; opts..., show_bands, resolution=opts_big.resolution)
    @saveplot accuracy_class_asin_scaled = Plots.plot_sigma(acc_s, ["Asin"]; opts..., show_bands, resolution=opts_big.resolution)
    @saveplot accuracy_class_asinnorm_scaled = Plots.plot_sigma(acc_s, ["AsinNorm"]; opts..., show_bands, resolution=opts_big.resolution)

    local kernels = ["Acos0", "Acos1", "Acos2"]

    @saveplot accuracy_class_acos_all_scaled = Plots.plot_sigma(acc_s, kernels; opts..., resolution=opts_big.resolution)

    @saveplot exec_cost = Plots.exec_dataset(data_ex3; resolution=(600, 500))
end

5 in plot_list && let
    @info "Experiment run 4 (svms4_inc) - Increasing subsample"
    df_sum_best_b32fm = let
        df = experiment_data("svms4_inc", scan_dirs)
        df.subsample = (df.subsample .|> a -> isnothing(a) ? 1.0 : a)
        Plots.summarize_best(df, [:kernel_cat, :dataset_cat, :subsample, :sigma_scaled], by=:measure_test)
    end

    sort!(df_sum_best_b32fm, order(:sigma_scaled))

    # df_sum_best_b32fm.subsample

    @saveplot nRMSE_bank32fm_sampling = Plots.plot_sigma_subsample(df_sum_best_b32fm; measure=:measurement, linkyaxes=false)
end

6 in plot_list && let
    @info "Kernel plots"
    @saveplot kernel_asin = Plots.plot_kernel(
        Utils.kernel_asin_normalized; fig_opts=(; resolution=(375, 200)),
        x=range(-1, 1, 201)
    )

    @saveplot kernel_asin_3d_sig0001 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e-3)
    @saveplot kernel_asin_3d_sig1 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+0)
    @saveplot kernel_asin_3d_sig1000 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+3)

    @saveplot kernel_acos0_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 0)
    @saveplot kernel_acos1_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 1)
    @saveplot kernel_acos2_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 2)
end

7 in plot_list && let
    @info "Heatmaps"
    heatmap_df = Plots.regression(Plots.data_nrmse_s())
    heatmap_df_class = Plots.classification(Plots.data_nrmse_s())
    local opts = (;
        alpha=0.0001,
        dims=(4, 8),
        vertical=false,
        opts_big...
    )
    @saveplot heatmaps_rbf_asinnorm = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, opts...)
    @saveplot heatmaps_rbf_asin = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Asin, opts...)
    @saveplot heatmaps_asin_asinnorm = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Asin, kernel_r=:AsinNorm, opts...)
    @saveplot heatmaps_rbf_acos1 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos1, opts...)

    @saveplot heatmaps_rbf_asinnorm_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, measure=:per_fold, opts...)
    @saveplot heatmaps_rbf_asin_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Asin, measure=:per_fold, opts...)
    @saveplot heatmaps_asin_asinnorm_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Asin, kernel_r=:AsinNorm, measure=:per_fold, opts...)
    # @saveplot heatmaps_rbf_acos1_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos1, measure=:per_fold, opts...)

    @saveplot heatmaps_asinnorm_asinnorm_ = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:AsinNorm, kernel_r=:AsinNorm, opts...)
    @saveplot heatmaps_asinnorm_asinnorm_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:AsinNorm, kernel_r=:AsinNorm, measure=:per_fold, opts...)

    # FIX: asinnorm against acos does not work.
    # @saveplot heatmaps_asinnorm_acos1 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Acos1, kernel_r=:AsinNorm, resolution=(2200, 1700))

    # TODO: acos0 and acos2 do not have data for this. Acos0 since it does not
    # have different values for sigma, and acos2 because it is too slow and many
    # executions are missing.
    # @saveplot heatmaps_rbf_acos0 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos0, resolution=(2200, 1700))
    # @saveplot heatmaps_rbf_acos2 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos2, resolution=(2200, 1700))

    @saveplot heatmaps_rbf_asinnorm_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, sigma=:sigma_scaled, opts...)
    @saveplot heatmaps_rbf_asin_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Asin, sigma=:sigma_scaled, opts...)
    @saveplot heatmaps_asin_asinnorm_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Asin, kernel_r=:AsinNorm, sigma=:sigma_scaled, opts...)
    @saveplot heatmaps_rbf_acos1_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos1, sigma=:sigma_scaled, opts...)

    # Classification

    @saveplot heatmaps_rbf_asinnorm_class_pvalues = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:RadialBasis, kernel_r=:AsinNorm, measure=:per_fold, opts...)
    @saveplot heatmaps_rbf_asin_class_pvalues = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:RadialBasis, kernel_r=:Asin, measure=:per_fold, opts...)
    # @saveplot heatmaps_asin_asinnorm_class_pvalues = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:Asin, kernel_r=:AsinNorm, measure=:per_fold, opts...)

    @saveplot heatmaps_rbf_asinnorm_class = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:RadialBasis, kernel_r=:AsinNorm, opts...)
    @saveplot heatmaps_rbf_asin_class = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:RadialBasis, kernel_r=:Asin, opts...)
    # @saveplot heatmaps_asin_asinnorm_class = Plots.plot_all_heatmaps(heatmap_df_class; kernel_l=:Asin, kernel_r=:AsinNorm, opts...)
end

@info "DONE in $(time() - start_time) seconds"
