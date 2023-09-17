#!/usr/bin/env julia

using TFM, TFM.Plots

using DataFramesMeta
using DrWatson

using CairoMakie

@info "Activating CairoMakie with LaTeX theme"
CairoMakie.activate!()
Plots.tex_theme!()
Plots.no_color_cycle!()

resolution = (1200, 800) # For big plots
show_bands = true
scan_dirs = "--scan" in ARGS
# WARN: --scan only affects Experiment 3, since 1 and 2 have already been
# purged.

@info scan_dirs ? "Scanning directories" : "Not scanning directories (pass --scan to scan for new data)"

@info "Experiment run 1 (svms/) MSE (Frenay grid)"
let
    local opts = (;
        linkxaxes=true, linkyaxes=false,
        show_rbf=true
    )
    local mse = @chain Plots.experiment_data("svms", false) begin
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
    local nrmse = @chain Plots.experiment_data("svms_2", false) begin
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
    local data_ex3 = Plots.experiment_data("svms3", scan_dirs)

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
