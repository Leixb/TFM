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

@info scan_dirs ? "Scanning directories" : "Not scanning directories (pass --scan to scan for new data)"

@info "Experiment run 1 (svms/) MSE (Frenay grid)"
let
    local opts = (;
        linkxaxes=true, linkyaxes=false,
        show_rbf=true
    )
    local mse = @chain Plots.experiment_data("svms", scan_dirs) begin
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
    local nrmse = @chain Plots.experiment_data("svms_2", scan_dirs) begin
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
    local nrmse_s = @chain Plots.experiment_data("svms3", scan_dirs) begin
        Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma])
        Plots.regression()
    end
    local opts = (;
        linkyaxes=true, linkxaxes=true,
        sigma=:sigma_scaled
    )

    @saveplot nRMSE_all_scaled = Plots.plot_sigma(nrmse_s; opts..., resolution)
    @saveplot nRMSE_frenay_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay); opts...)
    @saveplot nRMSE_frenay_s_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Small); opts...)
    @saveplot nRMSE_frenay_l_scaled = Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Large); opts...)

    @saveplot nRMSE_delve_bank_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 32; opts...)
    @saveplot nRMSE_delve_bank_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Bank, 8; opts...)
    @saveplot nRMSE_delve_pumadyn_32_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 32; opts...)
    @saveplot nRMSE_delve_pumadyn_8_scaled = Plots.plot_delve(nrmse_s, DataSets.Pumadyn, 8; opts...)
end

@info "Kernel plots"
let
    @saveplot kernel_asin = Plots.plot_kernel(Utils.kernel_asin_normalized)
    @saveplot kernel_asin_3d_sig0001 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e-3)
    @saveplot kernel_asin_3d_sig1 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+0)
    @saveplot kernel_asin_3d_sig1000 = Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+3)

    @saveplot kernel_acos0_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 0)
    @saveplot kernel_acos1_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 1)
    @saveplot kernel_acos2_3d = Plots.plot_kernel_3d(Utils.kernel_acos, 1, 2)
end
