import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def run_ridge_r2(df, feature_file, output_file, alphas):
    feature_names = pd.read_csv(feature_file)['feature'].tolist()
    X = df[feature_names]
    y = df['target']
    ids = df['ID']

    loo = LeaveOneOut()
    all_results = []
    train_r2_scores = {a: [] for a in alphas}
    y_true_all = []
    y_pred_by_alpha = {a: [] for a in alphas}

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_id = ids.iloc[test_index].values[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rec = {'test_id': test_id, 'y_true': y_test.iat[0]}

        for a in alphas:
            model = Ridge(alpha=a)
            model.fit(X_train_scaled, y_train)
            pred_test = model.predict(X_test_scaled)[0]
            pred_train = model.predict(X_train_scaled)

            rec[f'y_pred_{a}'] = pred_test
            train_r2_scores[a].append(r2_score(y_train, pred_train))
            y_pred_by_alpha[a].append(pred_test)

        y_true_all.append(y_test.iat[0])
        all_results.append(rec)

    results_df = pd.DataFrame(all_results)

    metrics_df = pd.DataFrame({
        'alpha': alphas,
        'log_alpha': np.log10(alphas),
        'R2_train': [sum(train_r2_scores[a]) / len(train_r2_scores[a]) for a in alphas],
        'R2_loocv': [r2_score(y_true_all, y_pred_by_alpha[a]) for a in alphas],
    })

    best_alpha_idx = metrics_df['R2_loocv'].idxmax()
    best_log_alpha = metrics_df.loc[best_alpha_idx, 'log_alpha']

    # Save metrics
    metrics_df.to_csv(output_file + "_r2_metrics.csv", index=False)

    # Plot with log(alpha)
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df['log_alpha'], metrics_df['R2_train'], color='blue', label='Training R²')
    plt.plot(metrics_df['log_alpha'], metrics_df['R2_loocv'], color='green', label='LOOCV R²')
    plt.axvline(x=best_log_alpha, color='red', linestyle='--', label='Best alpha for LOOCV')
    plt.title('Training vs LOOCV R² Across log(Alpha)')
    plt.xlabel('log(Alpha)')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file + "_r2_logalpha_plot.png")
    plt.show()


#------data---------
BASE_DIR = Path(__file__).parent

data = BASE_DIR / "Cleaned_data_imputed.csv"
feature_file_ari  = BASE_DIR / "ARI_features_for_ridgereg.csv"
feature_file_scared = BASE_DIR / "SCARED_features_for_ridgereg.csv"

alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

#------ARI---------
df = pd.read_csv(data)
df['target'] = df['ARI_6_P']
run_ridge_r2(df, feature_file_ari, "ARI_Ridge", alphas)

#------SCARED---------
df = pd.read_csv(data)
df['target'] = df['SCARED_P']
run_ridge_r2(df, feature_file_scared, "SCARED_Ridge", alphas)
