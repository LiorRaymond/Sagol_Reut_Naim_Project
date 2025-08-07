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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score

# —————————————————————————————————————————————————————
# PARAMETERS & PATHS
# —————————————————————————————————————————————————————

BASE_DIR = Path(__file__).parent

DATA_DIR    = BASE_DIR / "data"
cleaned_fp  = DATA_DIR / "Cleaned_data.csv"
vars_fp     = DATA_DIR / "variables_to_corroelation.csv"
tt_fp       = DATA_DIR / "variables_to_boxplots.csv"

out_dir     = BASE_DIR / "results"
out_dir.mkdir(exist_ok=True)

bad_et_ids = [23149, 24234]
alpha  = 0.10
threshold_vif = 10
seed = 42
random.seed(seed)

# —————————————————————————————————————————————————————
# 1. LOAD & PREPARE
# —————————————————————————————————————————————————————
def load_and_prepare_data():
    data = pd.read_csv(cleaned_fp)
    data = data[~data["ID"].isin(bad_et_ids)]
    data["Dx_binary"] = data["Dx"].apply(lambda x: 0 if x == "HV" else 1)

    tt_vars = pd.read_csv(tt_fp)["variable"]
    corr_vars = pd.read_csv(vars_fp)["variable"]
    tt_data = data[tt_vars].copy()
    features = data[corr_vars]
    return data, features, tt_data

# —————————————————————————————————————————————————————
# 2. HOLD-OUT SPLIT
# —————————————————————————————————————————————————————
def holdout_split(data, tt_data):

    X_train, X_test, y_train, y_test = train_test_split(tt_data, data["Dx_binary"], test_size=0.2,
    random_state=seed, stratify=data["Dx_binary"])
    return X_train, X_test, y_train, y_test

# ———sanity check for hold-out split
def ttest_holdout_split(X_train, X_test, y_train, y_test):
    train_df = pd.DataFrame(X_train, columns=X_train.columns)
    train_df["Dx_binary"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test, columns=X_test.columns)
    test_df["Dx_binary"] = y_test.reset_index(drop=True)

    results = []
    for col in train_df.select_dtypes(include=[np.number]).columns:
        if col != "Dx_binary":
            train_vals = train_df[col].dropna()
            test_vals = test_df[col].dropna()
            if len(train_vals) > 1 and len(test_vals) > 1:
                t_stat, p_val = ttest_ind(train_vals, test_vals, equal_var=False)
            else:
                t_stat, p_val = np.nan, np.nan
            results.append((col, t_stat, p_val))

    ttest_df = pd.DataFrame(results, columns=["Feature", "t_stat", "p_value"])
    rej, p_fdr = fdrcorrection(ttest_df["p_value"], alpha=alpha)
    ttest_df["p_fdr"] = p_fdr
    ttest_df["significant"] = rej
    ttest_df.to_csv(out_dir / "ttest_holdout_split.csv", index=False)
    print("significant differences between training and testing sets for features:")
    print(ttest_df[ttest_df["significant"]])
    return ttest_df



# —————————————————————————————————————————————————————
#3. FEATURE SELECTION
# —————————————————————————————————————————————————————
# Calculate correlations
def calculate_correlations(data, features, X_train, X_test, alpha):
    y_ari = data.loc[X_train.index, "ARI_6_P"]
    y_scared = data.loc[X_train.index, "SCARED_P"]

    r_ari, p_ari = [], []
    r_scared, p_scared = [], []
    features_list = features.columns.tolist()

    for col in features_list:
        r1, p1 = pearsonr(X_train[col], y_ari)
        r2, p2 = pearsonr(X_train[col], y_scared)

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

    feature_corr_df.to_csv(out_dir / "feature_correlation_ari_scared.csv", index=False)
    # Select features with significant correlation to either ARI or SCARED
    significant_features_df = feature_corr_df.query("significant_ARI or significant_SCARED")
    significant_features_df.to_csv(out_dir / "significant_features_only.csv", index=False)

    selected_features = significant_features_df["Feature"].tolist()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_selected.to_csv(out_dir / "X_train_final.csv", index=False)
    X_test_selected.to_csv(out_dir / "X_test_final.csv", index=False)

    return X_train_selected


# ---3.b VIF CALCULATION------
def vif(X):
    """Return DataFrame with VIF for each feature in X (must not contain NaNs)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_no_na = X.dropna()
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_no_na.columns
    vif_df["VIF"] = [variance_inflation_factor(X_no_na.values, i)
                     for i in range(X_no_na.shape[1])]
    return vif_df

def compute_vif(X_train_selected):
    X_vif_filtered = X_train_selected.copy()
    vif_result = vif(X_vif_filtered)
    vif_result.to_csv(out_dir / "vif_result.csv", index=False)
    return

#---3.c. correletion among selected features------
def correlation_between_features(X_train_selected, alpha):
    """Calculate pairwise Pearson correlation among features."""
    feature_pairs = []
    r_values = []
    p_values = []

    for col1, col2 in combinations(X_train_selected.columns, 2):
        r, p = pearsonr(X_train_selected[col1], X_train_selected[col2])
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
    return

def main():
    data, features, tt_data = load_and_prepare_data()
    X_train, X_test, y_train, y_test = holdout_split(data, tt_data)
    ttest_df = ttest_holdout_split(X_train, X_test, y_train, y_test)
    #feature selection
    X_train_selected = calculate_correlations(data, features, X_train, X_test, alpha)
    compute_vif(X_train_selected)
    correlation_between_features(X_train_selected, alpha)

main()

# —————————————————————————————————————————————————————
# 4. MODELING — LOOCV with Ridge Regression
# —————————————————————————————————————————————————————



