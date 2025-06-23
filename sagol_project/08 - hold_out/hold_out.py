import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import fdrcorrection
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
from pathlib import Path
import random
from scipy.stats import ttest_ind
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# —————————————————————————————————————————————————————
# PARAMETERS & PATHS
# —————————————————————————————————————————————————————

BASE_DIR = Path(__file__).parent

DATA_DIR    = BASE_DIR / "data"
cleaned_fp  = DATA_DIR / "Cleaned_data.csv"
vars_fp     = DATA_DIR / "variables_to_corroelation.csv"
et_fp       = DATA_DIR / "eye_tracking_variables.csv"

out_dir     = BASE_DIR / "results"
out_dir.mkdir(exist_ok=True)

bad_et_ids = [23149, 24234]
alpha  = 0.10
threshold_vif = 12
ff_cor_threshold = 0.9
seed = 42
random.seed(seed)

# —————————————————————————————————————————————————————
# 1. LOAD & PREPARE
# —————————————————————————————————————————————————————

data = pd.read_csv(cleaned_fp)
et_vars = pd.read_csv(et_fp)["variable"]
data.loc[data["ID"].isin(bad_et_ids), et_vars] = np.nan
data["Dx_binary"] = data["Dx"].apply(lambda x: 0 if x == "HV" else 1)


corr_vars = pd.read_csv(vars_fp)["variable"]
features = data[corr_vars]

# —————————————————————————————————————————————————————
# 2. HOLD-OUT SPLIT
# —————————————————————————————————————————————————————

stratify_col = 'Dx_binary'
X_train, X_test, y_train, y_test = train_test_split(
    features, data[stratify_col], test_size=0.2, random_state=seed, stratify=data[stratify_col]
)

# —————————————————————————————————————————————————————
#3. FEATURE SELECTION
# —————————————————————————————————————————————————————
#---3.a.Correlate each feature with ARI and SCARED-----
def masked_pearson(x, y):
    """Return (r, p) for x,y ignoring NaNs; or (nan, nan) if <2 points."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

# Extract ARI and SCARED for training participants only
y_ari = data.loc[X_train.index, "ARI_6_P"]
y_scared = data.loc[X_train.index, "SCARED_P"]

# Calculate correlations
r_ari, p_ari = [], []
r_scared, p_scared = [], []
features_list = X_train.columns

for col in features_list:
    r1, p1 = masked_pearson(X_train[col], y_ari)
    r2, p2 = masked_pearson(X_train[col], y_scared)

    r_ari.append(r1)
    p_ari.append(p1)
    r_scared.append(r2)
    p_scared.append(p2)

# FDR correction
rej_ari, p_fdr_ari = fdrcorrection(p_ari, alpha=alpha)
rej_scared, p_fdr_scared = fdrcorrection(p_scared, alpha=alpha)

# Combine results
feature_corr_df = pd.DataFrame({
    "Feature": features_list,
    "r_ARI": r_ari,
    "p_ARI": p_ari,
    "p_ARI_fdr": p_fdr_ari,
    "significant_ARI": rej_ari,
    "r_SCARED": r_scared,
    "p_SCARED": p_scared,
    "p_SCARED_fdr": p_fdr_scared,
    "significant_SCARED": rej_scared
})

# Save to file
feature_corr_df.to_csv(out_dir / "feature_selection_ari_scared.csv", index=False)

# Select features with significant correlation to either ARI or SCARED
selected_features = feature_corr_df.query("significant_ARI or significant_SCARED")["Feature"].tolist()
X_train_selected = X_train[selected_features]


# ---3.b VIF CALCULATION------
def compute_vif(X):
    """Return DataFrame with VIF for each feature in X (must not contain NaNs)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_no_na = X.dropna()
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_no_na.columns
    vif_df["VIF"] = [variance_inflation_factor(X_no_na.values, i)
                     for i in range(X_no_na.shape[1])]
    return vif_df

# Copy of selected features
X_vif_filtered = X_train_selected.copy()

# Iteratively remove feature with highest VIF > threshold
while True:
    vif_result = compute_vif(X_vif_filtered)
    max_row = vif_result.loc[vif_result["VIF"].idxmax()]
    if max_row["VIF"] > threshold_vif:
        X_vif_filtered = X_vif_filtered.drop(columns=[max_row["Feature"]])
    else:
        break

# Final VIF table (all VIFs <= threshold)
vif_result = compute_vif(X_vif_filtered)
vif_result.to_csv(out_dir / "vif_filtered.csv", index=False)

#---3.c. correletion among selected features------
feature_pairs = []
r_values = []
p_values = []

for col1, col2 in combinations(X_vif_filtered.columns, 2):
    r, p = masked_pearson(X_vif_filtered[col1], X_vif_filtered[col2])
    feature_pairs.append((col1, col2))
    r_values.append(r)
    p_values.append(p)

rej, p_fdr = fdrcorrection(p_values, alpha=alpha)

correlation_df = pd.DataFrame({
    "Feature_1": [pair[0] for pair in feature_pairs],
    "Feature_2": [pair[1] for pair in feature_pairs],
    "r": r_values,
    "p_uncorrected": p_values,
    "p_fdr": p_fdr,
    "significant": rej
})
correlation_df.to_csv(out_dir / "feature_correlation.csv", index=False)

high_corr_df = correlation_df.query("abs(r) > @ff_cor_threshold and significant")

selected_features = set()
to_remove = set()

# Create DataFrame for removed features with their r-values
removed_df = []

for _, row in high_corr_df.iterrows():
    f1, f2 = row["Feature_1"], row["Feature_2"]

    r1_ari = safe_pearson(X_train[f1], y_train[ari_col])
    r2_ari = safe_pearson(X_train[f2], y_train[ari_col])
    r1_scared = safe_pearson(X_train[f1], y_train[scared_col])
    r2_scared = safe_pearson(X_train[f2], y_train[scared_col])

    r1_avg = np.nanmean([abs(r1_ari), abs(r1_scared)])
    r2_avg = np.nanmean([abs(r2_ari), abs(r2_scared)])

    removed = f2 if r1_avg >= r2_avg else f1
    kept = f1 if removed == f2 else f2

    removed_df.append({
        "Feature_Removed": removed,
        "Feature_Kept": kept,
        "r_ARI_removed": r1_ari if removed == f1 else r2_ari,
        "r_SCARED_removed": r1_scared if removed == f1 else r2_scared,
        "r_avg_removed": r1_avg if removed == f1 else r2_avg,
        "Reason": f"Higher correlation retained for {kept}"
    })

removed_df = pd.DataFrame(removed_df)
removed_df.to_csv(out_dir / "removed_features.csv", index=False)

# ---3.d. Test if there are significant differences between the training and testing set for all features and the dependent variables of interest---
train_df = X_train.copy()
train_df["ARI_6_P"] = data.loc[X_train.index, "ARI_6_P"]
train_df["SCARED_P"] = data.loc[X_train.index, "SCARED_P"]

test_df = X_test.copy()
test_df["ARI_6_P"] = data.loc[X_test.index, "ARI_6_P"]
test_df["SCARED_P"] = data.loc[X_test.index, "SCARED_P"]

results = []
for col in train_df.columns:
    train_vals = train_df[col].dropna()
    test_vals = test_df[col].dropna()
    if len(train_vals) > 1 and len(test_vals) > 1:
        t_stat, p_val = ttest_ind(train_vals, test_vals, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
    results.append((col, t_stat, p_val))

ttest_df = pd.DataFrame(results, columns=["Feature", "t_stat", "p_value"])

rej, p_fdr = fdrcorrection(ttest_df["p_value"], alpha=0.10)
ttest_df["p_fdr"] = p_fdr
ttest_df["significant"] = rej
ttest_df.to_csv(out_dir / "ttest_results.csv", index=False)

if ttest_df["significant"].any():
    print("Significant differences found between training and testing sets for some features.")
    breakpoint()

# Final selected features after removing highly correlated ones
final_features = list(set(X_vif_filtered.columns) - to_remove)
X_train_final = X_train[final_features]
X_test_final = X_test[final_features]

X_train_final.to_csv(out_dir / "X_train_selected.csv", index=False)

#----3.e. Finalize selected features------
X_test_final = X_test[X_train_final.columns]

X_train_final.to_csv(out_dir / "X_train_final.csv", index=False)
X_test_final.to_csv(out_dir / "X_test_final.csv", index=False)

# —————————————————————————————————————————————————————
# 4. MODELING — LOOCV with Ridge Regression
# —————————————————————————————————————————————————————



