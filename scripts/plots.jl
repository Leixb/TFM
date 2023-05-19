#!/usr/bin/env julia

using TFM
using TFM.Plots

using DataFramesMeta
using DrWatson

using CairoMakie
CairoMakie.activate!()
Plots.tex_theme!()

const mse = @chain Plots.experiment_data("svms", false) begin
    @rsubset(:dataset isa DataSets.DataSet)
    Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma])
    @rename(:measure_test = :measurement)
    Plots.regression()
end

const nrmse = @chain Plots.experiment_data("svms_2", false) begin
    Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma], by=:measurement)
    Plots.regression()
end

const nrmse_s = @chain Plots.experiment_data("svms3", true) begin
    Plots.summarize_best([:kernel_cat, :dataset_cat, :sigma])
    Plots.regression()
end

const common_opts = (;
    linkxaxis=true,
    linkyaxis=false,
    show_rbf=true,
)

@saveplot MSE_all     Plots.plot_sigma(mse; show_bands=true, common_opts..., resolution=(1200, 800))
@saveplot MSE_frenay  Plots.plot_sigma(@rsubset(mse, :dataset isa DataSets.Frenay); show_bands=true, common_opts...)

@saveplot nRMSE_all     Plots.plot_sigma(nrmse; common_opts..., resolution=(1200, 800))
@saveplot nRMSE_frenay  Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Frenay); common_opts...)
@saveplot nRMSE_bank    Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Bank); common_opts...)
@saveplot nRMSE_pumadyn Plots.plot_sigma(@rsubset(nrmse, :dataset isa DataSets.Pumadyn); common_opts...)

@saveplot nRMSE_s_all     Plots.plot_sigma(nrmse_s; common_opts..., resolution=(1200, 800))
@saveplot nRMSE_s_frenay  Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Frenay); common_opts...)
@saveplot nRMSE_s_pumadyn Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Pumadyn); common_opts...)
@saveplot nRMSE_s_bank    Plots.plot_sigma(@rsubset(nrmse_s, :dataset isa DataSets.Bank); common_opts...)

@saveplot kernel_asin Plots.plot_asin()
@saveplot kernel_asin_3d_sig0001 Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e-3)
@saveplot kernel_asin_3d_sig1    Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+0)
@saveplot kernel_asin_3d_sig1000 Plots.plot_kernel_3d(Utils.kernel_asin_normalized, 1e+3)

@saveplot kernel_acos0_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 0)
@saveplot kernel_acos1_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 1)
@saveplot kernel_acos2_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 2)
