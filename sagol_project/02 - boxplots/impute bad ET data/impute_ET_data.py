import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load data
data = pd.read_csv("Cleaned_data.csv")
et_vars = pd.read_csv("eye_tracking_variables.csv")["variable"].tolist()
bad_ids = [23149, 24234]

# Loop over each eye-tracking variable
for var in et_vars:

    for pid in bad_ids:
        # Select DMDD group excluding bad_ids
        group_vals = data[(data["Dx"] == "DMDD") & (~data["ID"].isin(bad_ids))][var].dropna()
        
        # Compute skew and kurtosis
        sk = skew(group_vals)
        ku = kurtosis(group_vals, fisher=True)

        # Impute median if skewed/kurtotic, else mean
        impute_val = group_vals.median() if abs(sk) > 2 or abs(ku) > 2 else group_vals.mean()

        # Fill in the imputed value
        data.loc[data["ID"] == pid, var] = impute_val

# Save the updated dataset
data.to_csv("Cleaned_data_imputed.csv", index=False)

