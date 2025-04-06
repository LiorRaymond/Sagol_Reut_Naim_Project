import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import seaborn as sns


data = pd.read_csv("Cleaned_data.csv")

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
    sns.boxplot(x='Dx_binary', y=var, data=data)

    plt.title(f"T test of {var} by Dx_binary\nT-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

    plt.savefig(os.path.join(output_dir, f"{var}_T_test.png"))
    plt.close()

