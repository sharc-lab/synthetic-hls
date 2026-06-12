from pathlib import Path
from pprint import pp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms

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
COLORS_NEW = {
    # Region 1 – red hue (benchmarks train → synthetic test)
    'Polybench-Base':       '#FFADAD',
    'Machsuite-Base':       '#F69494',
    'Poly+Mach-Base':       '#ED7B7B',
    'Polybench-Final':      '#E56161',
    'Machsuite-Final':      '#DC4848',
    'Poly+Mach-Final':      '#D32F2F',

    # Region 2 – orange hue (cross on benchmarks)
    'Polybench-Machsuite':  '#FFD08A',
    'Machsuite-Polybench':  '#F4A261',

    # Region 3 – green hue (Base train → benchmark test)
    'Base-Polybench':       '#95E1B7',
    'Base-Machsuite':       '#6FCF97',
    'Base-Poly+Mach':       '#52B788',

    # Region 4 – teal hue (Final train → benchmark test)
    'Final-Polybench':      '#6ECFBF',
    'Final-Machsuite':      '#48B8A4',
    'Final-Poly+Mach':      '#2A9D8F',

    # Region 5 – blue hue (same data train → test)
    'Polybench-Polybench':  '#D0EAFF',
    'Machsuite-Machsuite':  '#A8D4FF',
    'Poly+Mach-Poly+Mach':  '#8BCAFF',
    'Base-Base':            '#6BB5F0',
    'Final-Final':          '#4EA0DB',

    # Region 6 – purple hue (cross eval synthetic)
    'Base-Final':           '#B070E0',
    'Final-Base':           '#8034C2',
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

MAP_Y_LABEL = {
    'Latency': 'Latency',
    'syn-LUT': '# LUTs',
    'syn-FF': '# FFs',
}

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

    ZORDER_BARS = 10
    LINEWIDTH_BARS = 0.0

    for i, (idx, row) in enumerate(data.iterrows()):
        ax.bar(x[i] - w/2, row[f"train_{metric_name}"], w,
               color=COLORS_NEW[row["Datasets(Train-Test)"]], alpha=0.55, edgecolor='white', linewidth=LINEWIDTH_BARS, zorder=ZORDER_BARS)
        ax.bar(x[i] + w/2, row[f"test_{metric_name}"], w,
               color=COLORS_NEW[row["Datasets(Train-Test)"]], alpha=1.0,  edgecolor='white', linewidth=LINEWIDTH_BARS, zorder=ZORDER_BARS)

    ymax = max(row[f"test_{metric_name}"] for _, row in data.iterrows())
    for i, (idx, row) in enumerate(data.iterrows()):
        tv = row[f"test_{metric_name}"]
        # if tv > ymax * 0.15:

        # ax.text(x[i] + w/2, tv * 1.08, f'{tv:.2f}', ha='center', va='bottom',
        #         fontsize=7, fontweight='bold', color="black")

        # put each text like before in a little white box
        ax.text(x[i] + w/2, tv * 1.3, f'{tv:.2f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color="black",
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.85, linewidth=0.5), zorder=10)
                


    FACE_ALPHA = 0.06
    EDGE_ALPHA = 0.3

    regions = [
        {"xmin": -0.5, "xmax": 5.5,  "color": "#E63946"},
        {"xmin": 5.5,  "xmax": 7.5,  "color": "#F4A261"},
        {"xmin": 7.5,  "xmax": 10.5, "color": "#52B788"},
        {"xmin": 10.5, "xmax": 13.5, "color": "#2A9D8F"},
        {"xmin": 13.5, "xmax": 18.5, "color": "#8bcaff"},
        {"xmin": 18.5, "xmax": 20.5, "color": "#8034c2"},
    ]

    for r in regions:
        rgb = mcolors.to_rgb(r["color"])
        ax.axvspan(
            r["xmin"], r["xmax"],
            facecolor=(*rgb, FACE_ALPHA),
            edgecolor=(*rgb, EDGE_ALPHA),
            linewidth=1,
            zorder=-2,
        )

    # for each region compure the avergae vlaue of the test bars in those regions and draw a horizontal line at that value only in the bounds for that region

    # regions = [
    #     (-0.5, 5.5),
    #     (5.5, 7.5),
    #     (7.5, 10.5),
    #     (10.5, 13.5),
    #     (13.5, 18.5),
    #     (18.5, 20.5),
    # ]
    regions = [(d["xmin"], d["xmax"]) for d in regions]

    for region in regions:
        start, end = region
        data_in_region = data[(data.index >= start) & (data.index < end)]
        average_value = data_in_region[f"test_{metric_name}"].mean().round(2)
        # geomean
        # values = data_in_region[f"test_{metric_name}"].values.astype(float)
        # average_value = np.exp(np.mean(np.log(values))).round(2)


        print(f"Average value in region {start} to {end}: {average_value}")
        ax.hlines(average_value, color='black', linestyle='--', linewidth=1.2, xmin=start, xmax=end)
        # ax.text(start + (end - start) / 2, average_value * 1.08, f'{average_value}', ha='center', va='bottom',
        #         fontsize=7, fontweight='bold', color="black")




    ax.set_xticks(x)
    # use the "Datasets(Train-Test)" column to get the labels
    ax.set_xticklabels([LABELS.get(row["Datasets(Train-Test)"], row["Datasets(Train-Test)"]) for _, row in data.iterrows()], rotation=35, ha='right', fontsize=9, color="black", fontweight='bold')
    ax.set_ylabel(f'RMSE - {metric_name}', fontsize=11)
    ax.set_title(f'QoR Model Cross-Validation Accuracy On Train-Test Dataset Pairs / {metric_name}',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_yscale('log')

    # set ylim to be 10% above the max value
    ax.set_ylim(None, max(row[f"test_{metric_name}"] for _, row in data.iterrows()) * 2.6)

    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)



    tr_patch = mpatches.Patch(facecolor='grey', alpha=0.5, label='Train RMSE', edgecolor='black', linewidth=0.8, linestyle='--')
    te_patch = mpatches.Patch(facecolor='grey', alpha=1.0, label='Test RMSE', edgecolor='black', linewidth=0.8, linestyle='-')
    avg_patch = plt.Line2D([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='Avg. RMSE in Set')
    ax.legend(handles=[tr_patch, te_patch, avg_patch], loc='upper center', fontsize=9, ncol=3, framealpha=1.0, columnspacing=0.75, facecolor='white')

    # ax.annotate('⚠ narrow-trained\n(fails to generalise)',
    #             xy=(5.5, 0.0), xycoords=('data', 'axes fraction'),
    #             xytext=(5.7, 0.92), textcoords=('data', 'axes fraction'),
    #             ha='left', va='top', fontsize=8, color='#E63946', fontstyle='italic',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                       edgecolor='#E63946', alpha=0.85, linewidth=1.2))

    ax.set_xlim(-0.5, num_cases - 0.5)

    plt.tight_layout()
    # fig.savefig(outpath, dpi=180, bbox_inches='tight')
    # plt.close()
    # print(f'Saved: {outpath}')
    return fig


def make_giant_bar_plot(metrics: list[str], df_clean: pd.DataFrame, order: list[str] | None = None) -> Figure:

    fig, axes = plt.subplots(len(metrics), 1, figsize=(13, 7))

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        # use the same code from make_bar_plot function, dont call it but repreat the same code
        data = df_clean.copy()
        if order is not None:
            data = data.set_index("Datasets(Train-Test)").loc[order].reset_index()

        base_columns = data.columns[data.columns.str.contains('Base')]
        columns_to_drop = []
        for col in base_columns:
            if col.startswith('test_') or col.startswith('train_'):
                if not col.endswith(metric):
                    columns_to_drop.append(col)
        data = data.drop(columns=columns_to_drop)

        num_cases = data["Datasets(Train-Test)"].nunique()
        x = np.arange(num_cases)
        w = 0.38

        LINEWIDTH_BARS = 0.0
        ZORDER_BARS = 10
        
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.bar(x[i] - w/2, row[f"train_{metric}"], w,
                   color=COLORS_NEW[row["Datasets(Train-Test)"]], alpha=0.55, edgecolor='white', linewidth=LINEWIDTH_BARS, zorder=ZORDER_BARS)
            ax.bar(x[i] + w/2, row[f"test_{metric}"], w,
                   color=COLORS_NEW[row["Datasets(Train-Test)"]], alpha=1.0,  edgecolor='white', linewidth=LINEWIDTH_BARS, zorder=ZORDER_BARS)

        ymax = max(row[f"test_{metric}"] for _, row in data.iterrows())
        for i, (idx, row) in enumerate(data.iterrows()):
            tv = row[f"test_{metric}"]

            ax.text(x[i] + w/2, tv * 1.3, f'{tv:.2f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color="black",
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='gray', alpha=0.85, linewidth=0.5), zorder=10)


        FACE_ALPHA = 0.06
        EDGE_ALPHA = 0.3

        regions = [
            {"xmin": -0.5, "xmax": 5.5,  "color": "#E63946"},
            {"xmin": 5.5,  "xmax": 7.5,  "color": "#F4A261"},
            {"xmin": 7.5,  "xmax": 10.5, "color": "#52B788"},
            {"xmin": 10.5, "xmax": 13.5, "color": "#2A9D8F"},
            {"xmin": 13.5, "xmax": 18.5, "color": "#8bcaff"},
            {"xmin": 18.5, "xmax": 20.5, "color": "#8034c2"},
        ]

        for r in regions:
            rgb = mcolors.to_rgb(r["color"])
            ax.axvspan(
                r["xmin"], r["xmax"],
                facecolor=(*rgb, FACE_ALPHA),
                edgecolor=(*rgb, EDGE_ALPHA),
                linewidth=1,
                zorder=-2,
            )
        




        regions = [(d["xmin"], d["xmax"]) for d in regions]

        for region in regions:
            start, end = region
            data_in_region = data[(data.index >= start) & (data.index < end)]
            average_value = data_in_region[f"test_{metric}"].mean().round(2)
    

            print(f"Average value in region {start} to {end}: {average_value}")
            ax.hlines(average_value, color='black', linestyle='--', linewidth=1.2, xmin=start, xmax=end)

        # for each region add a text box at the top and centered to that region that says what taht region

        REGION_LABELS = [
            "Train Bench / Test Synth",
            "Train & Test\nBench",
            "Train Synth (Base)\nTest Bench",
            "Train Synth (Final)\nTest Bench",
            "Train-Test Same Dataset",
            "Train &\nTest Synth"
        ]

        if metric_idx == 0:

            for region_idx, region in enumerate(regions):
                region_label = REGION_LABELS[region_idx]
                region_start, region_end = region
                trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(region_start + (region_end - region_start) / 2, 0.86, region_label, ha='center', va='center',
                        fontsize=8, fontweight='bold', color="black",
                        transform=trans,
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='gray', alpha=1.0, linewidth=0.5), zorder=10)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(row["Datasets(Train-Test)"], row["Datasets(Train-Test)"]) for _, row in data.iterrows()], rotation=35, ha='right', fontsize=9, color="black", fontweight='bold')
        
        ax.set_ylabel(f'RMSE: {MAP_Y_LABEL[metric]}', fontsize=11, fontweight='bold')
        ax.yaxis.set_label_coords(-0.045, 0.5)
        ax.set_yscale('log')

        top_scale = [25,3,2]

        ax.set_ylim(None, max(row[f"test_{metric}"] for _, row in data.iterrows()) * top_scale[metric_idx])
        ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)



        ax.set_xlim(-0.5, num_cases - 0.5)

        if metric_idx != len(metrics) - 1:
            ax.set_xticks([])
            # remove the tick marks
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)



    fig.suptitle('QoR Model Cross-Validation Error On Different Train-Test Dataset Pairs',
             fontsize=14, fontweight='bold', y=0.99)
    tr_patch = mpatches.Patch(facecolor='grey', alpha=0.5, label='Train RMSE', edgecolor='black', linewidth=0.8, linestyle='--')
    te_patch = mpatches.Patch(facecolor='grey', alpha=1.0, label='Test RMSE', edgecolor='black', linewidth=0.8, linestyle='-')
    avg_patch = plt.Line2D([0, 1], [0, 1], color='black', linestyle='--', linewidth=1, label='Avg. RMSE in Set')
    fig.legend(handles=[tr_patch, te_patch, avg_patch],
            loc='upper center', fontsize=9, ncol=3,
            framealpha=1.0, columnspacing=0.75, facecolor='white',
            bbox_to_anchor=(0.5, 0.96))
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=0.5)

    return fig


DIR_FIGURES = DIR_CURRENT / 'figures'
DIR_FIGURES.mkdir(parents=True, exist_ok=True)

# fig1 = make_bar_plot('Latency', df_clean, order=ORDER_CUSTOM)
# fig2 = make_bar_plot('syn-LUT', df_clean, order=ORDER_CUSTOM)
# fig3 = make_bar_plot('syn-FF',  df_clean, order=ORDER_CUSTOM)

# fig1.savefig(DIR_FIGURES / 'fig1_latency_train_test.png', dpi=300)
# fig2.savefig(DIR_FIGURES / 'fig2_synlut_train_test.png', dpi=300)
# fig3.savefig(DIR_FIGURES / 'fig3_synff_train_test.png', dpi=300)

fig4 = make_giant_bar_plot(['Latency', 'syn-LUT', 'syn-FF'], df_clean, order=ORDER_CUSTOM)
fig4.savefig(DIR_FIGURES / 'fig4_giant_bar_plot.png', dpi=300)

