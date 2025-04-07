import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns


data = pd.read_csv("Cleaned_data.csv")
id_col = "ID"

data["Dx_binary"] = data["Dx"].apply(lambda x: 0 if x == "HV" else 1)
cols = list(data.columns)
dx_index = cols.index("Dx")
cols.insert(dx_index + 1, cols.pop(cols.index("Dx_binary")))
data = data[cols]

variables = ["ARI_1_C",	"ARI_1_P",	"ARI_6_C",	"ARI_6_P",	"SCARED_C",	"SCARED_P"]

output_dir = "t_test_graphs"

for var in variables:
    group_0 = data[data["Dx_binary"] == 0][var]
    group_1 = data[data["Dx_binary"] == 1][var]
    
    t_stat, p_val = stats.ttest_ind(group_0, group_1, nan_policy='omit')

    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x='Dx_binary', y=var, data=data)
    for i, group in enumerate([0, 1]):
        group_data = data[data["Dx_binary"] == group][[id_col, var]].dropna()
        q1 = group_data[var].quantile(0.25)
        q3 = group_data[var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = group_data[(group_data[var] < lower_bound) | (group_data[var] > upper_bound)]

        y_offsets = {}
        y_range = group_data[var].max() - group_data[var].min()
        tolerance = y_range * 0.05
        offset_step = y_range * 0.05

        for _, row in outliers.iterrows():
            outlier_y = row[var]
            offset = 0
            for existing_y, existing_offset in y_offsets.items():
                if abs(outlier_y - existing_y) < tolerance:
                    offset = max(offset, existing_offset + offset_step)
            plt.text(i, outlier_y + offset, f"ID: {row[id_col]}", fontsize=8, verticalalignment='bottom')
            y_offsets[outlier_y] = offset

    plt.title(f"T test of {var} by Dx_binary\nT-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

    plt.savefig(os.path.join(output_dir, f"{var}_T_test.png"))
    plt.close()

