# pre-processing and evaluation batch RIM2D runs
setup for shahin's 2023-08-01 Ahr sims

2023-12-11: made some changes to better share w/ cnnf

## scripts
- _01asc_to_concat: load asc files into  xarray dataset
- _02clip: clip arrays to make evenly divisible
- _03resample: building resample of coarse array
- _04confusion: building confusion grids for each coarse sim (vs. fine)
- _05eval: evaluating coarse coarse wsh grids (vs. fine)
- dataAnalysis: plotting evaluation metrics:
    - plot_inun_perf_stack2: plot (2x) inundation performance metrics for each scenario as a function of Mannings
    - plot_stats_per_sim: compute WSH stats for each scenario as a function of Mannings
    
plot_inun_perf_stack2

![img](/img/inun_perf2_0_1494.png)

plot_stats_per_sim

![img](/img/stats_relplot_1495.png)
