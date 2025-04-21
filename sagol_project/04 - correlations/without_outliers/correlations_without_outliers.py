import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import fdrcorrection

# 1. Load and z‑score everything
data = pd.read_csv("Cleaned_data.csv")
vars_to_corr = pd.read_csv("variables_to_corroelation.csv")["variable"]
df_z = data[vars_to_corr].apply(zscore)

outliers = [24254, 24212]
data = data[~data['ID'].isin(outliers)]

# 2. Define features vs. outcomes
features = df_z.drop(columns=['ARI_6_P', 'SCARED_P'])
ari     = df_z['ARI_6_P']
scared  = df_z['SCARED_P']

# 3. Pairwise feature–feature correlations + p‑values
corr_matrix    = features.corr()
pval_matrix    = pd.DataFrame(index=features.columns, columns=features.columns, dtype=float)
for i in features.columns:
    for j in features.columns:
        _, p = pearsonr(features[i], features[j])
        pval_matrix.loc[i, j] = p

# 4. Feature–outcome correlations + p‑values
ari_corr = features.corrwith(ari)
ari_p    = pd.Series(index=features.columns, dtype=float)
for feat in features.columns:
    _, p = pearsonr(features[feat], ari)
    ari_p[feat] = p

scared_corr = features.corrwith(scared)
scared_p    = pd.Series(index=features.columns, dtype=float)
for feat in features.columns:
    _, p = pearsonr(features[feat], scared)
    scared_p[feat] = p

# 5. FDR correction across *all* tests
all_p = pd.concat([
    pval_matrix.stack(),
    ari_p.rename_axis("feature"), 
    scared_p.rename_axis("feature")
])

rejected, p_corrected = fdrcorrection(all_p.values, alpha=0.10)

# 6. Put corrected values back into the same shapes
n_ff = pval_matrix.size
ff_corr = p_corrected[:n_ff].reshape(pval_matrix.shape)
ari_corr_fdr    = p_corrected[n_ff : n_ff + len(ari_p)]
scared_corr_fdr = p_corrected[-len(scared_p):]

corrected_pval_matrix = pd.DataFrame(ff_corr, index=features.columns, columns=features.columns)
ari_p_fdr    = pd.Series(ari_corr_fdr,    index=features.columns)
scared_p_fdr = pd.Series(scared_corr_fdr, index=features.columns)

# 7. Save everything
corr_matrix.to_csv('correlation_matrix_without_outliers.csv')
pval_matrix.to_csv('p_values_matrix_without_outliers.csv')
corrected_pval_matrix.to_csv('corrected_p_values_matrix_without_outliers.csv')

ari_corr.to_csv('ari_correlation_without_outliers.csv')
ari_p.to_csv('ari_p_values_without_outliers.csv')
ari_p_fdr.to_csv('ari_corrected_p_values_without_outliers.csv')

scared_corr.to_csv('scared_correlation_without_outliers.csv')
scared_p.to_csv('scared_p_values_without_outliers.csv')
scared_p_fdr.to_csv('scared_corrected_p_values_without_outliers.csv')