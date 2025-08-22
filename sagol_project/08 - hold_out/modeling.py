import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from pathlib import Path

# Load Cleaned_data.csv for labels
data = pd.read_csv("data/Cleaned_data.csv")
data = data[~data["ID"].isin([23149, 24234])]  # Exclude bad IDs, as in preparing.py
data["Dx_binary"] = data["Dx"].apply(lambda x: 0 if x == "HV" else 1)

score_types = ["parent", "child"]
feature_modes = ["combined", "ari", "scared"]

for score_type in score_types:
    for mode in feature_modes:
        mode_dir = Path(f"results/{score_type}/{mode}")
        model_dir = Path(f"results/{score_type}_{mode}_model")
        model_dir.mkdir(exist_ok=True)

        X_train = pd.read_csv(mode_dir / "X_train_final.csv")
        X_test = pd.read_csv(mode_dir / "X_test_final.csv")

        y_train = data[data["ID"].isin(X_train["ID"])]["Dx_binary"].reset_index(drop=True)
        y_test = data[data["ID"].isin(X_test["ID"])]["Dx_binary"].reset_index(drop=True)

        alphas = [0.1, 0.3, 0.5, 0.7, 1.0]
        loocv_records = []

        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(X_train):
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
            test_id = X_te["ID"].values[0]

            X_tr_nofeatures = X_tr.drop(columns=["ID"])
            X_te_nofeatures = X_te.drop(columns=["ID"])
            if X_tr_nofeatures.shape[1] == 0:
                print(f"No features selected for {score_type} mode '{mode}'. Skipping modeling.")
                break

            imputer = SimpleImputer(strategy="mean")
            scaler = StandardScaler(with_mean=False)
            X_tr_imp = imputer.fit_transform(X_tr_nofeatures)
            X_te_imp = imputer.transform(X_te_nofeatures)
            X_tr_scaled = scaler.fit_transform(X_tr_imp)
            X_te_scaled = scaler.transform(X_te_imp)

            rec = {"test_id": test_id, "y_true": y_te.values[0]}
            for alpha in alphas:
                model = RidgeClassifier(alpha=alpha)
                model.fit(X_tr_scaled, y_tr)
                pred = model.predict(X_te_scaled)[0]
                rec[f"y_pred_{alpha}"] = pred
            loocv_records.append(rec)

        loocv_df = pd.DataFrame(loocv_records)
        loocv_df.to_csv(model_dir / "holdout_LOOCV_results.txt", sep="\t", index=False)

        # Skip metrics if no predictions were made
        if loocv_df.empty or not any(f"y_pred_{alpha}" in loocv_df.columns for alpha in alphas):
            print(f"No LOOCV predictions for {score_type} mode '{mode}'. Skipping metrics and test evaluation.")
            continue

        metrics = []
        for alpha in alphas:
            preds = loocv_df[f"y_pred_{alpha}"]
            bal_acc = balanced_accuracy_score(loocv_df["y_true"], preds)
            acc = (loocv_df["y_true"] == preds).mean()
            f1 = f1_score(loocv_df["y_true"], preds)
            metrics.append({"alpha": alpha, "accuracy": acc, "balanced_accuracy": bal_acc, "f1": f1})

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(model_dir / "holdout_overall_metrics.txt", sep="\t", index=False)

        with open(model_dir / "holdout_classification_reports.txt", "w") as f:
            for alpha in alphas:
                preds = loocv_df[f"y_pred_{alpha}"]
                f.write(f"Classification Report for alpha={alpha}:\n")
                f.write(classification_report(loocv_df["y_true"], preds, zero_division=0))
                f.write("\n\n")

        best = metrics_df.loc[metrics_df["balanced_accuracy"].idxmax(), "alpha"]

        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_train_imp = imputer.fit_transform(X_train.drop(columns=["ID"]))
        X_test_imp = imputer.transform(X_test.drop(columns=["ID"]))
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        final_model = RidgeClassifier(alpha=best)
        final_model.fit(X_train_scaled, y_train)
        test_preds = final_model.predict(X_test_scaled)

        test_results = pd.DataFrame({
            "ID": X_test["ID"],
            "y_true": y_test,
            "y_pred": test_preds
        })
        test_results.to_csv(model_dir / "holdout_test_predictions.txt", sep="\t", index=False)

        with open(model_dir / "holdout_test_summary.txt", "w") as f:
            f.write(f"Best alpha: {best}\n")
            f.write(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, test_preds):.3f}\n")
            f.write(f"Test F1 Score: {f1_score(y_test, test_preds):.3f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test, test_preds, zero_division=0))