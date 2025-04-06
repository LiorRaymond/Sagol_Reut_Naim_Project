import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("Cleaned_data.csv")
variables_to_plot = pd.read_csv("variables_to_boxplots.csv")["variable"].tolist()

output_dir = "boxplots_graphs"

for var in variables_to_plot:
    if var in data.columns:
        plt.figure(figsize=(8, 6))
        data.boxplot(column=var)
        plt.title(f'Boxplot of {var}')
        plt.savefig(os.path.join(output_dir, f"{var}_boxplot.png"))
        plt.close()
    else:
        print(f"Warning: {var} not found in data columns")
