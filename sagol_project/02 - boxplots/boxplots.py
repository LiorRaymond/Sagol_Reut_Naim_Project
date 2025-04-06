import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("Cleaned_data.csv")

variables_to_plot = pd.read_csv("variables_to_boxplots.csv")["variable"].tolist()

output_dir = "boxplots_graphs"

for var in variables_to_plot:
    if var in data.columns:
        file_name = f"{var}_boxplot.png"
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            continue

        subset = data[[var]].dropna()
        n = len(subset)
        
        median = subset[var].median()
        std = subset[var].std()

        plt.figure(figsize=(8, 6))
        ax = data.boxplot(column=var)
        stats_text = f'n = {n}   |   median = {median:.2f}   |   std = {std:.2f}'
        plt.text(x=0.5, y=1.10, s=stats_text,
                    transform=ax.transAxes,  
                    ha='right', va='bottom',
                    fontsize=10, color='black')

        plt.title(f'Boxplot of {var}')
        plt.savefig(os.path.join(output_dir, f"{var}_boxplot.png"))
        plt.close()
    else:
        print(f"Warning: {var} not found in data columns")
