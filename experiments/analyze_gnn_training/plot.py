from pathlib import Path
from pprint import pp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Load data directly from Excel ────────────────────────────────────────────
DIR_CURRENT = Path(__file__).parent
df = pd.read_excel(DIR_CURRENT / 'Results_RMSE_Train-Test_4096.xlsx', header=None)

# show all columns
# enable printing all rows
pd.set_option('display.max_columns', None)
print(df)

# Row 3 = column headers, rows 4+ = data; cols 2=label, 3-7=train, 8-12=test
col_names = list(df.iloc[3, 3:8])   # ['Latency','syn-LUT','syn-FF','syn-DSP','syn-BRAM']
data_rows = df.iloc[4:, :]

df_clean = df.copy()
df_clean = df_clean.iloc[3:] # remove the first 3 rows
df_clean = df_clean.iloc[:, 2:] # remove first two columns
df_clean = df_clean.reset_index(drop=True)

# make the first row the column names
df_clean.columns = df_clean.iloc[0].values
df_clean = df_clean.iloc[1:]
df_clean = df_clean.reset_index(drop=True)

# Datasets(Train-Test) Latency syn-LUT  syn-FF syn-DSP syn-BRAM   Latency  \
# 0      Polybench-Final  0.4743  0.0373  0.0374   0.041   0.0016  190.8231   
# 1       Polybench-Base  0.4743  0.0373  0.0374   0.041   0.0016  141.6436   
# 2      Machsuite-Final   0.489  0.1753  0.0755  0.1958   0.0253  106.9253   
# 3       Machsuite-Base   0.489  0.1753  0.0755  0.1958   0.0253   47.0575   
# 4      Poly+Mach-Final  0.6438  0.1473  0.0943  0.1549   0.0192   53.2345   

#   syn-LUT  syn-FF syn-DSP syn-BRAM  
# 0  1.6594  0.4427   3.551   1.1324  
# 1  5.7533  0.1387     4.8   0.5695  
# 2  7.2639  0.3426  1.6505   1.6266  
# 3  3.3172  0.2095   0.787   0.6558  
# 4  3.7244  0.5895  0.8908   0.1487

# prefix the first set of columns (latency, syn-LUT, syn-FF, syn-DSP, syn-BRAM) with 'train_' and the second set of columns with 'test_'
# dont mess with the first column
df_clean.columns = [df_clean.columns[0]] + ['train_' + col for col in df_clean.columns[1:5]] + ['test_' + col for col in df_clean.columns[5:]]


df_clean['dataset_name_train'] = df_clean['Datasets(Train-Test)'].str.split('-').str[0].str.strip()
df_clean['dataset_name_test'] = df_clean['Datasets(Train-Test)'].str.split('-').str[1].str.strip()

print(df_clean.head())

# keys   = list(data_rows.iloc[:, 2])

# train  = {r[2]: list(r[3:8].astype(float))  for _, r in data_rows.iterrows()}
# test   = {r[2]: list(r[8:13].astype(float)) for _, r in data_rows.iterrows()}
# metrics = col_names

# pp(train)
# pp(test)

pd.reset_option('display.max_columns')



# ── Colours & labels ──────────────────────────────────────────────────────────
COLORS = {
    'Polybench-Final':      '#E63946',
    'Polybench-Base':       '#FF6B6B',
    'Machsuite-Final':      '#C1121F',
    'Machsuite-Base':       '#FF8FA3',
    'Poly+Mach-Final':      '#E07A5F',
    'Poly+Mach-Base':       '#F2CC8F',
    'Base-Machsuite':       '#2A9D8F',
    'Final-Machsuite':      '#264653',
    'Base-Poly+Mach':       '#52B788',
    'Final-Poly+Mach':      '#1B4332',
    'Base-Polybench':       '#74C69D',
    'Final-Polybench':      '#40916C',
    'Base-Final':           '#F4A261',
    'Final-Base':           '#E9C46A',
    'Polybench-Machsuite':  '#FFBA08',
    'Machsuite-Polybench':  '#E5A400',
    'Polybench-Polybench':  '#A8DADC',
    'Machsuite-Machsuite':  '#6FAFB8',
    'Poly+Mach-Poly+Mach':  '#9DC8C8',
    'Base-Base':            '#457B9D',
    'Final-Final':          '#1D3557',
}
LABELS = {
    'Polybench-Final':      'Poly → Final',
    'Polybench-Base':       'Poly → Base',
    'Machsuite-Final':      'Mach → Final',
    'Machsuite-Base':       'Mach → Base',
    'Poly+Mach-Final':      'Poly+Mach → Final',
    'Poly+Mach-Base':       'Poly+Mach → Base',
    'Base-Machsuite':       'Base → Mach',
    'Final-Machsuite':      'Final → Mach',
    'Base-Poly+Mach':       'Base → Poly+Mach',
    'Final-Poly+Mach':      'Final → Poly+Mach',
    'Base-Polybench':       'Base → Poly',
    'Final-Polybench':      'Final → Poly',
    'Base-Final':           'Base → Final',
    'Final-Base':           'Final → Base',
    'Polybench-Machsuite':  'Poly → Mach',
    'Machsuite-Polybench':  'Mach → Poly',
    'Polybench-Polybench':  'Poly → Poly',
    'Machsuite-Machsuite':  'Mach → Mach',
    'Poly+Mach-Poly+Mach':  'Poly+Mach → Poly+Mach',
    'Base-Base':            'Base → Base',
    'Final-Final':          'Final → Final',
}

ORDER_ORIGINAL = [
    'Polybench-Final',
    'Polybench-Base',
    'Machsuite-Final',
    'Machsuite-Base',
    'Poly+Mach-Final',
    'Poly+Mach-Base',
    'Base-Machsuite',
    'Final-Machsuite',
    'Base-Poly+Mach',
    'Final-Poly+Mach',
    'Base-Polybench',
    'Final-Polybench',
    'Base-Final',
    'Final-Base',
    'Polybench-Machsuite',
    'Machsuite-Polybench',
    'Polybench-Polybench',
    'Machsuite-Machsuite',
    'Poly+Mach-Poly+Mach',
    'Base-Base',
    'Final-Final',
]

ORDER_CUSTOM = [
    # benchmakrs train -> synthetic test
    'Polybench-Base',
    'Machsuite-Base',
    'Poly+Mach-Base',
    'Polybench-Final',
    'Machsuite-Final',
    'Poly+Mach-Final',

    #corss on benchmarks train -> test
    'Polybench-Machsuite',
    'Machsuite-Polybench',



    # now start with base and final as  train and benchmark test
    'Base-Polybench',
    'Base-Machsuite',
    'Base-Poly+Mach',
    'Final-Polybench',
    'Final-Machsuite',
    'Final-Poly+Mach',



    # pairs of same benchmarks train -> test
    'Polybench-Polybench',
    'Machsuite-Machsuite',
    'Poly+Mach-Poly+Mach',

    # pairs of same synthetic datasets train -> test
    'Base-Base',
    'Final-Final',

    # base final paris train -> test
    'Base-Final',
    'Final-Base',

]

# ── Plot function ─────────────────────────────────────────────────────────────
def make_bar_plot(metric_name, df_clean, order=None) -> Figure:

    data = df_clean.copy()
    
    if order is not None:
        data = data.set_index("Datasets(Train-Test)").loc[order].reset_index()


    base_columns = data.columns[data.columns.str.contains('Base')]
    columns_to_drop = []
    for col in base_columns:
        if col.startswith('test_') or col.startswith('train_'):
            if not col.endswith(metric_name):
                columns_to_drop.append(col)
    data = data.drop(columns=columns_to_drop)


    num_cases = data["Datasets(Train-Test)"].nunique()

    x = np.arange(num_cases)
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 3.5))
    # fig.patch.set_facecolor('#F8F9FA')
    # ax.set_facecolor('#F8F9FA')

    # ax.bar(x - w/2, [train[k][metric_idx] for k in keys], w,
    #        color=[COLORS[k] for k in keys], alpha=0.55, edgecolor='white', linewidth=0.8)
    # ax.bar(x + w/2, [test[k][metric_idx]  for k in keys], w,
    #        color=[COLORS[k] for k in keys], alpha=1.0,  edgecolor='white', linewidth=0.8)
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.bar(x[i] - w/2, row[f"train_{metric_name}"], w,
               color=COLORS[row["Datasets(Train-Test)"]], alpha=0.55, edgecolor='white', linewidth=0.8)
        ax.bar(x[i] + w/2, row[f"test_{metric_name}"], w,
               color=COLORS[row["Datasets(Train-Test)"]], alpha=1.0,  edgecolor='white', linewidth=0.8)

    ymax = max(row[f"test_{metric_name}"] for _, row in data.iterrows())
    for i, (idx, row) in enumerate(data.iterrows()):
        tv = row[f"test_{metric_name}"]
        # if tv > ymax * 0.15:
        ax.text(x[i] + w/2, tv * 1.08, f'{tv:.2f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color="black")
                
    # for the regiion where we train on benchmarks and eval on synthetic
    ax.axvspan(-0.5, 5.5, alpha=0.06, color='#E63946', zorder=-2)

    # for the regiion where we train on benchmakr and test on different benchmarks, should be light yellow ( alittle darker)
    ax.axvspan(5.5, 7.5, alpha=0.06, color='#F4A261', zorder=-2)

    # region where we tain on synthetic BASE and test on benchmarks, should be light green
    ax.axvspan(7.5, 10.5, alpha=0.06, color='#52B788', zorder=-2)

    # region where we tain on synthetic FINAL and test on benchmarks, should be a slightly darker green
    ax.axvspan(10.5, 13.5, alpha=0.06, color='#2A9D8F', zorder=-2)

    # region where we tain and test on the same data, should be light blue brighter 
    ax.axvspan(13.5, 18.5, alpha=0.06, color='#8bcaff', zorder=-2)

    # region where we crros eval the synthetic datasets, should be  purple brighter
    ax.axvspan(18.5, 20.5, alpha=0.08, color='#8034c2', zorder=-2)

    # for each region compure the avergae vlaue of the test bars in those regions and draw a horizontal line at that value only in the bounds for that region

    regions = [
        (-0.5, 5.5),
        (5.5, 7.5),
        (7.5, 10.5),
        (10.5, 13.5),
        (13.5, 18.5),
        (18.5, 20.5),
    ]

    for region in regions:
        start, end = region
        data_in_region = data[(data.index >= start) & (data.index < end)]
        # average_value = data_in_region[f"test_{metric_name}"].mean().round(2)
        # geomean
        values = data_in_region[f"test_{metric_name}"].values.astype(float)
        average_value = np.exp(np.mean(np.log(values))).round(2)


        print(f"Average value in region {start} to {end}: {average_value}")
        ax.hlines(average_value, color='black', linestyle='--', linewidth=1.2, xmin=start, xmax=end)
        # ax.text(start + (end - start) / 2, average_value * 1.08, f'{average_value}', ha='center', va='bottom',
        #         fontsize=7, fontweight='bold', color="black")




    ax.set_xticks(x)
    # use the "Datasets(Train-Test)" column to get the labels
    ax.set_xticklabels([LABELS.get(row["Datasets(Train-Test)"], row["Datasets(Train-Test)"]) for _, row in data.iterrows()], rotation=35, ha='right', fontsize=9, color="black", fontweight='bold')
    ax.set_ylabel(f'RMSE - {metric_name}', fontsize=11)
    ax.set_title(f'QoR Model Cross-Validation Accuracy On Different Train-Test Dataset Pairs ({metric_name}) (lower is better)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_yscale('log')

    # set ylim to be 10% above the max value
    ax.set_ylim(None, max(row[f"test_{metric_name}"] for _, row in data.iterrows()) * 2)

    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)



    tr_patch = mpatches.Patch(facecolor='grey', alpha=0.5, label='Train RMSE', edgecolor='black', linewidth=0.8, linestyle='--')
    te_patch = mpatches.Patch(facecolor='grey', alpha=1.0, label='Test RMSE', edgecolor='black', linewidth=0.8, linestyle='-')
    ax.legend(handles=[tr_patch, te_patch], loc='upper right', fontsize=9, framealpha=0.7)

    ax.annotate('⚠ narrow-trained\n(fails to generalise)',
                xy=(5.5, 0.0), xycoords=('data', 'axes fraction'),
                xytext=(5.7, 0.92), textcoords=('data', 'axes fraction'),
                ha='left', va='top', fontsize=8, color='#E63946', fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#E63946', alpha=0.85, linewidth=1.2))

    ax.set_xlim(-0.5, num_cases - 0.5)

    plt.tight_layout()
    # fig.savefig(outpath, dpi=180, bbox_inches='tight')
    # plt.close()
    # print(f'Saved: {outpath}')
    return fig

DIR_FIGURES = DIR_CURRENT / 'figures'
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

fig1 = make_bar_plot('Latency', df_clean, order=ORDER_CUSTOM)
fig2 = make_bar_plot('syn-LUT', df_clean, order=ORDER_CUSTOM)
fig3 = make_bar_plot('syn-FF',  df_clean, order=ORDER_CUSTOM)

fig1.savefig(DIR_FIGURES / 'fig1_latency_train_test.png', dpi=300)
fig2.savefig(DIR_FIGURES / 'fig2_synlut_train_test.png', dpi=300)
fig3.savefig(DIR_FIGURES / 'fig3_synff_train_test.png', dpi=300)


