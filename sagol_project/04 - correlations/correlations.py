import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path

# —————————————————————————————————————————————————————
# HELPER FUNCTIONS
# —————————————————————————————————————————————————————

def masked_pearson(x, y):
    """Return (r, p) for x,y ignoring NaNs; or (nan, nan) if <2 points."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])

def compute_matrices(df):
    """Compute corr and pval matrices for all columns of df."""
    cols = df.columns
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)
    pval = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            r, p = masked_pearson(df[ci], df[cj])
            corr.at[ci, cj]  = r
            pval.at[ci, cj] = p
    return corr, pval

# —————————————————————————————————————————————————————
# PARAMETERS & PATHS
# —————————————————————————————————————————————————————

BASE_DIR = Path(__file__).parent

DATA_DIR    = BASE_DIR / "data"
cleaned_fp  = DATA_DIR / "Cleaned_data.csv"
vars_fp     = DATA_DIR / "variables_to_corroelation.csv"
et_fp       = DATA_DIR / "eye_tracking_variables.csv"
out_dir     = BASE_DIR / "results"

out_dir = Path("./results")
out_dir.mkdir(exist_ok=True)

bad_et_ids = [23149, 24234]
alpha_fdr  = 0.10

# —————————————————————————————————————————————————————
# 1. LOAD & PREPARE
# —————————————————————————————————————————————————————

data = pd.read_csv(cleaned_fp)
et_vars = pd.read_csv(et_fp)["variable"]
data.loc[data["ID"].isin(bad_et_ids), et_vars] = np.nan

corr_vars = pd.read_csv(vars_fp)["variable"]
df_z = data[corr_vars].apply(lambda col: zscore(col, nan_policy='omit'))

features = df_z
ari      = df_z['ARI_6_P']
scared   = df_z['SCARED_P']

# —————————————————————————————————————————————————————
# 2. COMPUTE MATRICES
# —————————————————————————————————————————————————————

corr_ff, pval_ff = compute_matrices(features)

# feature–outcome
ari_r = features.apply(lambda col: masked_pearson(col, ari)[0])
ari_p = features.apply(lambda col: masked_pearson(col, ari)[1])

sc_r  = features.apply(lambda col: masked_pearson(col, scared)[0])
sc_p  = features.apply(lambda col: masked_pearson(col, scared)[1])

# —————————————————————————————————————————————————————
# 3. FLATTEN ALL TESTS INTO ONE TABLE
# —————————————————————————————————————————————————————

rows = []
cols = features.columns.tolist()

# 3a. feature–feature (upper triangle only)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        rows.append({
            "Type": "feature-feature",
            "Feature_1": cols[i],
            "Feature_2": cols[j],
            "r": corr_ff.iat[i, j],
            "p_uncorrected": pval_ff.iat[i, j]
        })

# 3b. feature–ARI
for feat in cols:
    rows.append({
        "Type": "feature-ARI",
        "Feature_1": feat,
        "Feature_2": "",
        "r": ari_r[feat],
        "p_uncorrected": ari_p[feat]
    })

# 3c. feature–SCARED
for feat in cols:
    rows.append({
        "Type": "feature-SCARED",
        "Feature_1": feat,
        "Feature_2": "",
        "r": sc_r[feat],
        "p_uncorrected": sc_p[feat]
    })

all_tests_df = pd.DataFrame(rows)

# —————————————————————————————————————————————————————
# 4. FDR CORRECTION
# —————————————————————————————————————————————————————

rej, p_corr = fdrcorrection(all_tests_df["p_uncorrected"].values, alpha=alpha_fdr)
all_tests_df["p_fdr"]       = p_corr
all_tests_df["significant"] = rej

# —————————————————————————————————————————————————————
# 4.5. EXPORT FDR-CORRECTED P-VALUES TO CSV 
# —————————————————————————————————————————————————————

# feature–feature
ff_fdr_long = all_tests_df.query("Type == 'feature-feature'")[["Feature_1", "Feature_2", "p_fdr"]]
ff_fdr_matrix = ff_fdr_long.pivot(index="Feature_1", columns="Feature_2", values="p_fdr")

features = ff_fdr_matrix.columns.union(ff_fdr_matrix.index)
ff_fdr_matrix = ff_fdr_matrix.reindex(index=features, columns=features)
ff_fdr_matrix = ff_fdr_matrix.combine_first(ff_fdr_matrix.T)

ff_fdr_matrix.to_csv(out_dir / "p_fdr_feature_feature_matrix.csv")

# feature–ARI
ari_fdr = all_tests_df.query("Type == 'feature-ARI'")[["Feature_1", "p_fdr"]]
ari_fdr.to_csv(out_dir / "p_fdr_feature_ARI.csv", index=False)

# feature–SCARED
sc_fdr = all_tests_df.query("Type == 'feature-SCARED'")[["Feature_1", "p_fdr"]]
sc_fdr.to_csv(out_dir / "p_fdr_feature_SCARED.csv", index=False)


# —————————————————————————————————————————————————————
# 5. SAVE RESULTS
# —————————————————————————————————————————————————————

# unified: only significant rows
sig_df = all_tests_df[all_tests_df["significant"]].copy()
sig_df = sig_df.drop(columns=["p_uncorrected"])
sig_df.to_csv(out_dir / "all_significant_correlations.csv", index=False)

# per-type significant
for t in ["feature-feature", "feature-ARI", "feature-SCARED"]:
    df_sub = sig_df.query("Type == @t")
    fn = f"significant_{t.replace('-', '_')}.csv"
    df_sub.to_csv(out_dir / fn, index=False)

# full matrices
corr_ff.to_csv(out_dir / "correlation_matrix.csv")

# feature–ARI and SCARED (with FDR)
sig_df.query("Type=='feature-ARI'")[['Feature_1','r','p_fdr']].to_csv(
    out_dir / "features_vs_ARI.csv", index=False)

sig_df.query("Type=='feature-SCARED'")[['Feature_1','r','p_fdr']].to_csv(
    out_dir / "features_vs_SCARED.csv", index=False)