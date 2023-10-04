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

resolution = (1200, 800) # For big plots
show_bands = true
if !@isdefined scan_dirs
    scan_dirs = "--scan" in ARGS || "-s" in ARGS
end
# WARN: --scan only affects Experiment 3, since 1 and 2 have already been
# purged.

@info scan_dirs ? "Scanning directories" : "Not scanning directories (pass --scan to scan for new data)"

@info "Benchmarks"
let
    @saveplot benchmark_time_inst = Plots.plot_benchmark_time_instances()
    @saveplot benchmark_time_improvement_old = Plots.exec_improvement()
end

@info "Experiment run 1 (svms/) MSE (Frenay grid)"
let
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
    @saveplot MSE_all = Plots.plot_sigma(mse; opts..., show_bands, resolution)
    @saveplot MSE_frenay = Plots.plot_sigma(@rsubset(mse, :dataset isa DataSets.Frenay); opts..., show_bands)
end

@info "Experiment run 2 (svms_2/) Normalized RMSE"
let
    local opts = (;
        linkxaxes=true, linkyaxes=false,
        show_rbf=true
    )
    local nrmse = @chain experiment_data("svms_2", false) begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma], by=:measurement)
        Plots.regression()
    end
    @saveplot nRMSE_all = Plots.plot_sigma(nrmse; opts..., resolution)
    @saveplot nRMSE_frenay = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Frenay); opts...)
    @saveplot nRMSE_frenay_s = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Small); opts...)
    @saveplot nRMSE_frenay_l = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Large); opts...)
    @saveplot nRMSE_bank = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Bank); opts...)
    @saveplot nRMSE_pumadyn = Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Pumadyn); opts...)
end

@info "Experiment run 3 (svms3/) nRMSE with scaled sigma"
let
    local data_ex3 = experiment_data("svms3", scan_dirs)

    local nrmse_s = @chain data_ex3 begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma])
        Plots.regression()
    end
    local opts = (;
        linkyaxes=true, linkxaxes=true,
        sigma=:sigma_scaled,
        show_rbf=true
    )

    @info "Experiment run 3 (smvs3/) - part 1: Regression plots"

    @saveplot nRMSE_all_scaled = Plots.plot_sigma(nrmse_s; opts..., show_bands, resolution)
    @saveplot nRMSE_frenay_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay); opts..., show_bands)
    @saveplot nRMSE_frenay_s_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Small); opts..., show_bands)
    @saveplot nRMSE_frenay_l_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Large); opts..., show_bands)

    @saveplot nRMSE_delve_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32; opts...)
    @saveplot nRMSE_delve_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8; opts...)
    @saveplot nRMSE_delve_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32; opts...)
    @saveplot nRMSE_delve_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8; opts...)

    local kernels = ["Acos0", "Acos1", "Acos2"]

    @saveplot nRMSE_acos_all_scaled = Plots.plot_sigma(nrmse_s, kernels; opts..., resolution)
    @saveplot nRMSE_acos_frenay_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay), kernels; opts...)
    @saveplot nRMSE_acos_frenay_s_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Small), kernels; opts...)
    @saveplot nRMSE_acos_frenay_l_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Large), kernels; opts...)

    @saveplot nRMSE_acos_delve_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32, kernels; opts...)
    @saveplot nRMSE_acos_delve_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8, kernels; opts...)
    @saveplot nRMSE_acos_delve_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32, kernels; opts...)
    @saveplot nRMSE_acos_delve_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8, kernels; opts...)

    @info "Experiment run 3 (smvs3/) - part 2: Classification plots"

    local acc_s = @chain data_ex3 begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma]; by=:measurement, maximum=true)
        Plots.classification()
    end

    @saveplot accuracy_class_all_scaled = Plots.plot_sigma(acc_s; opts..., show_bands, resolution)

    local kernels = ["Acos0", "Acos1", "Acos2"]

    @saveplot accuracy_class_acos_all_scaled = Plots.plot_sigma(acc_s, kernels; opts..., resolution)
end

@info "Experiment run 4 (svms4_inc) - Increasing subsample"
let
    df_sum_best_b32fm = let
        df = experiment_data("svms4_inc", scan_dirs)
        df.subsample = (df.subsample .|> a -> isnothing(a) ? 1.0 : a)
        Plots.summarize_best(df, [:kernel_cat, :dataset_cat, :subsample, :sigma_scaled], by=:measure_test)
    end

    sort!(df_sum_best_b32fm, order(:sigma_scaled))

    # df_sum_best_b32fm.subsample

    @saveplot nRMSE_bank32fm_sampling = Plots.plot_sigma_subsample(df_sum_best_b32fm; measure=:measurement, linkyaxes=false)
end

@info "Kernel plots"
let
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

@info "Heatmaps"
let
    heatmap_df = Plots.data_nrmse_s()
    @saveplot heatmaps_rbf_asinnorm = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, resolution=(2200, 1700))
    @saveplot heatmaps_rbf_asin = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Asin, resolution=(2200, 1700))
    @saveplot heatmaps_asin_asinnorm = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Asin, kernel_r=:AsinNorm, resolution=(2200, 1700))
    @saveplot heatmaps_rbf_acos1 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos1, resolution=(2200, 1700))

    @saveplot heatmaps_rbf_asinnorm_pvalues = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, measure=:per_fold, resolution=(2200, 1700))

    # FIX: asinnorm against acos does not work.
    # @saveplot heatmaps_asinnorm_acos1 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Acos1, kernel_r=:AsinNorm, resolution=(2200, 1700))

    # TODO: acos0 and acos2 do not have data for this. Acos0 since it does not
    # have different values for sigma, and acos2 because it is too slow and many
    # executions are missing.
    # @saveplot heatmaps_rbf_acos0 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos0, resolution=(2200, 1700))
    # @saveplot heatmaps_rbf_acos2 = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos2, resolution=(2200, 1700))

    @saveplot heatmaps_rbf_asinnorm_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:AsinNorm, sigma=:sigma_scaled, resolution=(2200, 1700))
    @saveplot heatmaps_rbf_asin_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Asin, sigma=:sigma_scaled, resolution=(2200, 1700))
    @saveplot heatmaps_asin_asinnorm_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:Asin, kernel_r=:AsinNorm, sigma=:sigma_scaled, resolution=(2200, 1700))
    @saveplot heatmaps_rbf_acos1_s = Plots.plot_all_heatmaps(heatmap_df; kernel_l=:RadialBasis, kernel_r=:Acos1, sigma=:sigma_scaled, resolution=(2200, 1700))
end

@info "DONE in $(time() - start_time) seconds"
