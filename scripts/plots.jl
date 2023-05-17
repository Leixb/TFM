using TFM
using TFM.Plots

using DataFramesMeta
using DrWatson

using CairoMakie
CairoMakie.activate!()
Plots.tex_theme!()

const df = Plots.experiment_data("svms_2", false)

const df_sum = Plots.summarize_best(df, [:kernel_cat, :dataset_cat, :sigma])

const df_reg = Plots.regression(df_sum)

const common_opts = (;
    linkxaxis=true,
    linkyaxis=false,
)

@saveplot nRMSE_all     Plots.plot_sigma(df_reg; common_opts..., resolution=(1200, 800))
@saveplot nRMSE_frenay  Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Frenay); common_opts...)
@saveplot nRMSE_bank    Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Bank); common_opts...)
@saveplot nRMSE_pumadyn Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Pumadyn); common_opts...)

@saveplot kernel_asin Plots.plot_asin()
@saveplot kernel_asin_3d Plots.plot_kernel_3d(Utils.kernel_asin)

@saveplot kernel_acos0_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 0)
@saveplot kernel_acos1_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 1)
@saveplot kernel_acos2_3d Plots.plot_kernel_3d(Utils.kernel_acos, 1, 2)
