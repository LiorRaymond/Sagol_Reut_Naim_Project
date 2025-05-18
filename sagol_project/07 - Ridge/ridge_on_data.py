#importa packages
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report


def run_ridge_regression(df, feature_file, output_file):
    # Load feature names
    feature_names = pd.read_csv(feature_file)['feature'].tolist()

    # Prepare data
    X = df[feature_names]
    y = df['target']
    ids = df['ID']

    # Define the set of alphas to compare
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    #initialize LOOCV
    loo = LeaveOneOut()

    #initialize results list
    all_results = []

    for train_index, test_index in loo.split(X):
        #split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_id = ids.iloc[test_index].values[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #prepare one record for this test sample
        rec = {'test_id': test_id,
            'y_true': y_test.iat[0]
            }

        #train and predict
        for a in alphas:
            clf = RidgeClassifier(alpha=a, class_weight='balanced')
            clf.fit(X_train_scaled, y_train)
            pred = clf.predict(X_test_scaled)[0]
            rec[f'y_pred_{a}'] = pred

        all_results.append(rec)

    #convert results list to a dataframe
    results_df = pd.DataFrame(all_results)

    #building matrix of overall accuracy & balanced accuracy by a
    metrics_by_alpha = {}
    for a in alphas:
        preds = results_df[f'y_pred_{a}']
        metrics_by_alpha[a] = {
            'accuracy':          accuracy_score(results_df['y_true'], preds),
            'balanced_accuracy': balanced_accuracy_score(results_df['y_true'], preds)
        }

    metrics_df = (
        pd.DataFrame.from_dict(metrics_by_alpha, orient='index')
          .rename_axis('alpha')
          .reset_index()
    )

    # Save all metrics to a text file
    with open(output_file + "_results.txt", "w") as f:
        f.write("LOOCV Results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        #overall accuracy & balanced accuracy by a
        f.write("Overall & Balanced Accuracy by alpha:\n")
        f.write(metrics_df.to_string(index=False, float_format="%.3f"))
        f.write("\n\n")

        for a in metrics_df['alpha']:
            preds = results_df[f'y_pred_{a}']
            f.write(f"Classification Report for alpha={a}:\n")
            f.write(classification_report(
                results_df['y_true'], preds, zero_division=0
            ))
            f.write("\n")


#loading data
df = pd.read_csv("Cleaned_data_imputed.csv")

#adding target column
df['target'] = df['Dx'].apply(lambda x: 0 if x == 'HV' else 1)

#running functions (no '.csv' in output_file)
run_ridge_regression(df, "ARI_features_for_ridgereg.csv", "ARI_Ridge")
run_ridge_regression(df, "SCARED_features_for_ridgereg.csv", "SCARED_Ridge")