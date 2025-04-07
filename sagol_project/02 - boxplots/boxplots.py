import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv("Cleaned_data.csv")
variables_to_plot = pd.read_csv("variables_to_boxplots.csv")["variable"].tolist()
output_dir = "boxplots_graphs"
id_col = "ID"

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

        #Adding IDs to ouliers
        outliers = ax.lines[6].get_ydata()
        outlier_ids = data.loc[data[var].isin(outliers), id_col].tolist()  
        outlier_x = [1] * len(outliers)

        y_offsets = {}
        y_range = plt.ylim()[1] - plt.ylim()[0]
        tolerance = y_range * 0.01 
        offset_step = y_range * 0.02
        for i, outlier in enumerate(outliers):
            offset = 0
            for existing_y, existing_offset in y_offsets.items():
                if abs(outlier - existing_y) < tolerance:
                    offset = max(offset, existing_offset + offset_step)
            plt.text(outlier_x[i], outlier + offset, f"ID: {outlier_ids[i]}", fontsize=8, verticalalignment='bottom')
            y_offsets[outlier] = offset


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
