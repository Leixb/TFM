using TFM
using TFM.Plots

using DataFramesMeta
using DrWatson

df = Plots.experiment_data("svms_2", false)

df_sum = Plots.summarize_best(df, [:kernel_cat, :dataset_cat, :sigma])

df_reg = Plots.regression(df_sum)

common_opts = (;
    linkxaxis=true,
    linkyaxis=false,
)

all = Plots.plot_sigma(df_reg; common_opts..., resolution=(1200, 800))

frenay  = Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Frenay); common_opts...)
bank    = Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Bank); common_opts...)
pumadyn = Plots.plot_sigma(@rsubset(df_reg, :dataset isa DataSets.Pumadyn); common_opts...)

save(plotsdir("nRMSE.pdf"), all)
save(plotsdir("nRMSE_frenay.pdf"), frenay)
save(plotsdir("nRMSE_bank.pdf"), bank)
save(plotsdir("nRMSE_pumadyn.pdf"), pumadyn)
