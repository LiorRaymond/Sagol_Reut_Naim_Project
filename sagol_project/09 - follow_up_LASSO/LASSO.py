import os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# 0) PATHS & CONFIG
# -------------------------
BASE_DIR    = Path(__file__).parent                              # .../09 - follow_up_LASSO
SOURCE_DIR  = BASE_DIR.parent / "08 - hold_out"                  # .../08 - hold_out
DATA_CSV    = SOURCE_DIR / "data" / "Cleaned_data.csv"
RESULTS_DIR = SOURCE_DIR / "results"                             # where X_train_final.csv / X_test_final.csv live

OUT_DIR = BASE_DIR / "results_l1_logistic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAD_IDS   = {23149, 24234}
LABEL_COL = "Dx_binary"   # 0=HV, 1=Clinical

# use inverse of alpha for C in LogisticRegression
# (RidgeClassifier uses alpha directly, but LogisticRegression uses C = 1/alpha)
ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]
def alpha_to_C(a): return 1.0 / a

score_types   = ["parent", "child"]
feature_modes = ["combined", "ari", "scared"]

# -------------------------
# 1) LOAD LABELS
# -------------------------
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_CSV}")

data = pd.read_csv(DATA_CSV)
data = data[~data["ID"].isin(BAD_IDS)].copy()
if "ID" not in data.columns or "Dx" not in data.columns:
    raise ValueError("Expected columns 'ID' and 'Dx' in Cleaned_data.csv")

data[LABEL_COL] = (data["Dx"] != "HV").astype(int)

# -------------------------
# 2) MAIN
# -------------------------
for score_type in score_types:
    for mode in feature_modes:
        mode_dir  = RESULTS_DIR / score_type / mode
        model_dir = OUT_DIR / f"{score_type}_{mode}_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        xtr = mode_dir / "X_train_final.csv"
        xte = mode_dir / "X_test_final.csv"
        if not xtr.exists() or not xte.exists():
            print(f"[WARN] Missing prepared files for {score_type}/{mode}: {xtr} | {xte}")
            continue

        X_train = pd.read_csv(xtr)
        X_test  = pd.read_csv(xte)
        if "ID" not in X_train.columns or "ID" not in X_test.columns:
            print(f"[WARN] No 'ID' column for {score_type}/{mode}. Skipping.")
            continue

        y_train = data.set_index("ID").loc[X_train["ID"], LABEL_COL].reset_index(drop=True)
        y_test  = data.set_index("ID").loc[X_test["ID"],  LABEL_COL].reset_index(drop=True)

        feat_cols = [c for c in X_train.columns if c != "ID"]
        if not feat_cols:
            print(f"[WARN] No features for {score_type}/{mode}. Skipping.")
            continue

        # -------------------------
        # LOOCV 
        # -------------------------
        loo = LeaveOneOut()
        loocv_records = []

        for tr_idx, te_idx in loo.split(X_train):
            X_tr = X_train.iloc[tr_idx][feat_cols].copy()
            X_te = X_train.iloc[te_idx][feat_cols].copy()
            y_tr = y_train.iloc[tr_idx].values
            y_te = int(y_train.iloc[te_idx].values[0])
            test_id = int(X_train.iloc[te_idx]["ID"].values[0])

            imputer = IterativeImputer(random_state=42)
            scaler  = StandardScaler()

            X_tr_imp = imputer.fit_transform(X_tr)
            X_te_imp = imputer.transform(X_te)

            X_tr_sc = scaler.fit_transform(X_tr_imp)
            X_te_sc = scaler.transform(X_te_imp)

            rec = {"test_id": test_id, "y_true": y_te}
            for a in ALPHAS:
                C = alpha_to_C(a)
                clf = LogisticRegression(
                    penalty="l1",
                    solver="liblinear", 
                    C=C,
                    max_iter=10000,
                    random_state=42
                )
                clf.fit(X_tr_sc, y_tr)
                
                proba = float(clf.predict_proba(X_te_sc)[:, 1][0])
                pred  = int(proba >= 0.5)

                rec[f"y_pred_{a}"]  = pred
                rec[f"proba_{a}"]   = proba
            loocv_records.append(rec)

        loocv_df = pd.DataFrame(loocv_records)
        loocv_df.to_csv(model_dir / "holdout_LOOCV_results.txt", sep="\t", index=False)

        if loocv_df.empty:
            print(f"[WARN] No LOOCV rows for {score_type}/{mode}.")
            continue

        # -------------------------
        # LOOCV METRICS & REPORTS
        # -------------------------
        metrics = []
        reports_txt = []
        cms_txt = []
        for a in ALPHAS:
            y_true = loocv_df["y_true"].values
            y_pred = loocv_df[f"y_pred_{a}"].values

            bal_acc = balanced_accuracy_score(y_true, y_pred)
            acc     = (y_true == y_pred).mean()
            f1m     = f1_score(y_true, y_pred, average="macro", zero_division=0)

            metrics.append({"alpha": a, "accuracy": acc, "balanced_accuracy": bal_acc, "f1_macro": f1m})

            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            cm_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])
            rep = classification_report(y_true, y_pred, zero_division=0,
                                        target_names=["Healthy (0)","Clinical (1)"])

            reports_txt.append(f"Classification Report for alpha={a}:\n{rep}\n")
            cms_txt.append(f"Confusion Matrix for alpha={a}:\n{cm_df}\n")

        metrics_df = pd.DataFrame(metrics).sort_values(by=["balanced_accuracy","f1_macro","accuracy"], ascending=False)
        metrics_df.to_csv(model_dir / "holdout_overall_metrics.txt", sep="\t", index=False)

        with open(model_dir / "holdout_classification_reports.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(reports_txt))
        with open(model_dir / "holdout_confusion_matrices.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(cms_txt))

        # Alpha with best balanced accuracy (tie-breaker: f1_macro, then accuracy)
        best_alpha = float(metrics_df.iloc[0]["alpha"])
        best_C = alpha_to_C(best_alpha)

        # -------------------------
        # FINAL MODEL on FULL TRAIN, TEST on HOLDOUT
        # -------------------------
        imputer_f = IterativeImputer(random_state=42)
        scaler_f  = StandardScaler()

        Xtr_imp = imputer_f.fit_transform(X_train[feat_cols])
        Xte_imp = imputer_f.transform(X_test[feat_cols])

        Xtr_sc = scaler_f.fit_transform(Xtr_imp)
        Xte_sc = scaler_f.transform(Xte_imp)

        final_clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=best_C,
            max_iter=10000,
            random_state=42
        )
        final_clf.fit(Xtr_sc, y_train.values)

        test_proba = final_clf.predict_proba(Xte_sc)[:, 1]
        test_pred  = (test_proba >= 0.5).astype(int)

        coef = final_clf.coef_.ravel()
        intercept = final_clf.intercept_[0]
        feature_names = feat_cols


        # Save coefficients to ensure they're preserved
        coeff_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef
        })
        coeff_df["abs_coefficient"] = np.abs(coeff_df["coefficient"])
        coeff_df = coeff_df.sort_values("abs_coefficient", ascending=False)
        coeff_df.to_csv(model_dir / "holdout_model_coefficients.txt", sep="\t", index=False)

        test_results = pd.DataFrame({
            "ID": X_test["ID"].values,
            "y_true": y_test.values,
            "y_pred": test_pred
        })
        test_results.to_csv(model_dir / "holdout_test_predictions.txt", sep="\t", index=False)

        conf_matrix = confusion_matrix(y_test.values, test_pred, labels=[0,1])
        conf_matrix_df = pd.DataFrame(conf_matrix, index=["True_Negative","True_Positive"],
                                      columns=["Pred_Negative","Pred_Positive"])
        conf_matrix_df.to_csv(model_dir / "holdout_test_confusion_matrix.txt", sep="\t")

        with open(model_dir / "holdout_test_summary.txt", "w") as f:
            f.write(f"Best alpha: {best_alpha}\n")
            f.write(f"C used: {best_C}\n")
            f.write(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test.values, test_pred):.3f}\n")
            f.write(f"Test F1 Macro: {f1_score(y_test.values, test_pred, average='macro', zero_division=0):.3f}\n")
            f.write("\nClassification Report:\n")
            f.write(classification_report(y_test.values, test_pred, zero_division=0))
