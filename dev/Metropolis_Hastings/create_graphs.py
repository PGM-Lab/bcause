import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import ast
import os

# Define paths
download_path = "/Users/antoniogonzalezalves/Documents/prueba_mh/"
pd.set_option('display.max_columns', None)

# ---------------------------------------------------------
# GLOBAL STYLE SETTINGS
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.5)

# Read the final results
df = pd.read_csv(os.path.join(download_path, "Final_Merged_All_Methods_3.csv"))


# Helper to parse string intervals
def parse_interval(s):
    try:
        if isinstance(s, (list, tuple)):
            return s
        if isinstance(s, str):
            return ast.literal_eval(s)
        return [np.nan, np.nan]
    except:
        return [np.nan, np.nan]


# 1. Define Algorithms dynamically
possible_algos = [
    'Gibbs_Sampling', 'Metropolis_Hastings',
    'Metropolis_Hastings_Exclude_Outliers',
    'Metropolis_Hastings_Zanella',
     'Metropolis_Hastings_Zanella_wo_outliers',
"Metropolis_Hastings_Parallel_Tempering",
    'Metropolis_Hastings_Parallel_Tempering_wo_outliers']
# 'Metropolis_Hastings_Swandsen_Wang', 'Metropolis_Hastings_AlwaysTrue', Metropolis_Hastings_Parallel_Tempering
# 'Metropolis_Hastings_Zanella'
algorithms = [algo for algo in possible_algos if algo in df.columns]
exact_col = 'Exact_Probability'

# 2. Parse Columns
cols_to_parse = algorithms + [exact_col]
for col in cols_to_parse:
    df[col + '_parsed'] = df[col].apply(parse_interval)

# Extract Exact Bounds ONCE (much faster than calculating it inside the loop every time)
df['ex_l'] = df[exact_col + '_parsed'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else np.nan)
df['ex_u'] = df[exact_col + '_parsed'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else np.nan)

# 3. Calculate RMSE and Coverage
for algo in algorithms:
    # Extract Bounds
    app_l = df[algo + '_parsed'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else np.nan)
    app_u = df[algo + '_parsed'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else np.nan)

    # RMSE
    mse = ((app_l - df['ex_l']) ** 2 + (app_u - df['ex_u']) ** 2) / 2
    df[algo + '_RMSE'] = np.sqrt(mse)


    # Coverage
    # Note: passing algo_name=algo fixes Python's late-binding loop issue!
    def calc_cov(row, algo_name=algo):
        ex = row[exact_col + '_parsed']
        app = row[algo_name + '_parsed']
        if not isinstance(ex, (list, tuple)) or not isinstance(app, (list, tuple)):
            return np.nan
        ex_l, ex_u = ex[0], ex[1]
        app_l, app_u = app[0], app[1]

        ex_len = ex_u - ex_l
        int_l = max(ex_l, app_l)
        int_u = min(ex_u, app_u)
        int_len = max(0, int_u - int_l)

        if ex_len > 1e-9:
            return (int_len / ex_len) * 100.0
        else:
            if app_l <= ex_l and app_u >= ex_u:
                return 100.0
            else:
                return 0.0


    df[algo + '_Coverage'] = df.apply(calc_cov, axis=1)

# 4. Filters
# filter by nparents = 2
# df = df[df['nparents'] == 3]

# Delete point estimation
# df['Exact_Probability_list'] = df['Exact_Probability'].apply(ast.literal_eval)
# df = df[df['Exact_Probability_list'].str[0] != df['Exact_Probability_list'].str[1]]
# Delete 0-coverage
# df = df[df['Metropolis_Hastings_Coverage'] > 0]

print(df[["Model_Index", "nparents", "nzr", "zdr", "cardinality"]].drop_duplicates())

# ---------------------------------------------------------
# PLOTTING WITH SEABORN
# ---------------------------------------------------------

# --- Dynamic Colors ---
# Automatically creates a color mapping for however many algorithms you actually ran!
palette = sns.color_palette("tab10", len(algorithms))
colors = {algo: palette[i] for i, algo in enumerate(algorithms)}

# --- Data Preparation (Melting) ---

# 1. Melt RMSE
rmse_cols = [algo + '_RMSE' for algo in algorithms]
df_rmse = df.melt(id_vars=['Iteration'], value_vars=rmse_cols, var_name='Algorithm', value_name='RMSE')
df_rmse['Algorithm'] = df_rmse['Algorithm'].str.replace('_RMSE', '')

# 2. Melt Coverage
cov_cols = [algo + '_Coverage' for algo in algorithms]
df_cov = df.melt(id_vars=['Iteration'], value_vars=cov_cols, var_name='Algorithm', value_name='Coverage')
df_cov['Algorithm'] = df_cov['Algorithm'].str.replace('_Coverage', '')

# 3. Melt Time dynamically
time_mapping = {}
for algo in algorithms:
    # Checks for dynamically named columns from the merge script
    if f'Time_{algo}' in df.columns:
        time_mapping[f'Time_{algo}'] = algo
    # Fallbacks for older column names
    elif algo == 'Gibbs_Sampling' and 'Time_gibbs' in df.columns:
        time_mapping['Time_gibbs'] = algo
    elif algo == 'Metropolis_Hastings' and 'Time_mh' in df.columns:
        time_mapping['Time_mh'] = algo

df_time = pd.DataFrame()
if time_mapping:
    temp_df = df[['Iteration'] + list(time_mapping.keys())].rename(columns=time_mapping)
    df_time = temp_df.melt(id_vars=['Iteration'], value_vars=list(time_mapping.values()), var_name='Algorithm',
                           value_name='Time')

# Helper for X-ticks
max_iter = df['Iteration'].max()
xticks_range = np.arange(0, max_iter + 1, 1000)

# ---------------------------------------------------------
# PLOT 1: Mean RMSE
# ---------------------------------------------------------
plt.figure(figsize=(10, 9))
sns.lineplot(
    data=df_rmse,
    x='Iteration',
    y='RMSE',
    hue='Algorithm',
    style='Algorithm',
    palette=colors,
    markers=True,
    dashes=False,
    errorbar=None,
    linewidth=4,
    markersize=14
)
plt.xticks(xticks_range)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
plt.title('RMSE Comparison for Causal Queries', fontsize=26, fontweight='bold', pad=20)
plt.xlabel('Iteration', fontsize=20, labelpad=15)
plt.ylabel('RMSE', fontsize=20, labelpad=15)
plt.legend(title='Algorithm', title_fontsize=20, fontsize=18, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# PLOT 2: Average Coverage Ratio
# ---------------------------------------------------------
plt.figure(figsize=(10, 9))
sns.lineplot(
    data=df_cov,
    x='Iteration',
    y='Coverage',
    hue='Algorithm',
    style='Algorithm',
    palette=colors,
    markers=True,
    dashes=False,
    errorbar=None,
    linewidth=4,
    markersize=14
)
plt.xticks(xticks_range)
plt.title('Average Coverage Ratio (%)', fontsize=26, fontweight='bold', pad=20)
plt.xlabel('Iteration', fontsize=20, labelpad=15)
plt.ylabel('Coverage (%)', fontsize=20, labelpad=15)
plt.legend(title='Algorithm', title_fontsize=20, fontsize=16, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# PLOT 3: Cumulative Learning Time
# ---------------------------------------------------------
if not df_time.empty:
    plt.figure(figsize=(10, 9))
    sns.lineplot(
        data=df_time,
        x='Iteration',
        y='Time',
        hue='Algorithm',
        style='Algorithm',
        palette=colors,
        markers=True,
        dashes=False,
        errorbar=None,
        linewidth=4,
        markersize=14
    )
    plt.xticks(xticks_range)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.title('Cumulative Learning Time (seconds)', fontsize=26, fontweight='bold', pad=20)
    plt.xlabel('Iteration', fontsize=20, labelpad=15)
    plt.ylabel('Time (s)', fontsize=20, labelpad=15)
    plt.legend(title='Algorithm', title_fontsize=20, fontsize=16, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# PLOT 4: Boxplot
# ---------------------------------------------------------
target_iterations = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
df_boxplot = df_rmse[df_rmse['Iteration'].isin(target_iterations)]

plt.figure(figsize=(10, 9))
sns.boxplot(
    data=df_boxplot,
    x='Iteration',
    y='RMSE',
    hue='Algorithm',
    palette=colors,
    showfliers=False,
    linewidth=2.5
)
plt.title('RMSE Distribution Across Algorithms', fontsize=24, fontweight='bold', pad=20)
plt.xlabel('Iteration Step', fontsize=18, labelpad=15)
plt.ylabel('RMSE', fontsize=18, labelpad=15)
plt.legend(title='Algorithm', title_fontsize=18, fontsize=16, loc='best')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()