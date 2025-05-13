#importa packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report


def run_logistic_regression(df, feature_file, output_file):
    # Load feature names
    feature_names = pd.read_csv(feature_file)['feature'].tolist()

    # Prepare data
    X = df[feature_names]
    y = df['target']
    ids = df['ID']

    #initialize LOOCV and LogReg
    loo = LeaveOneOut()
    model = LogisticRegression(solver='liblinear', class_weight='balanced')

    #initialize results list
    results = []

    for train_index, test_index in loo.split(X):
        #split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        test_id = ids.iloc[test_index].values[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        #train and predict
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

        #save results
        results.append({
            'test_id': test_id,
            'y_true': y_test.values[0],
            'y_pred': pred[0]
        })

    #convert results list to a dataframe
    results_df = pd.DataFrame(results)

    # Save all metrics to a text file
    with open(output_file + "_results.txt", "w") as f:
        f.write("LOOCV Results:\n")
        f.write(results_df.to_string(index=False))

        #overall accuracy
        overall_accuracy = accuracy_score(results_df['y_true'], results_df['y_pred'])
        f.write("\n\nOverall Accuracy:\n")
        f.write(str(overall_accuracy))

        #balanced accuracy
        balanced_accuracy = balanced_accuracy_score(results_df['y_true'], results_df['y_pred'])
        f.write("\n\nBalanced Accuracy:\n")
        f.write(str(balanced_accuracy))

        #zero division error handling included
        classification_rep = classification_report(results_df['y_true'], results_df['y_pred'], zero_division=0)
        f.write("\n\nClassification Report:\n")
        f.write(classification_rep)

        #manual per-class accuracy: accuracy for each class based on exact matches between y_true and y_pred
        class_accuracy = (
            results_df.groupby('y_true')[['y_true', 'y_pred']]
            .apply(lambda g: (g['y_true'] == g['y_pred']).mean())
            .to_dict()
        )
        f.write("\n\nManual Per-class Accuracy:\n")
        f.write(str(class_accuracy))

        #manual calcualtion of balanaced accuracy: average of recalls (true positive rates) for each class
        recalls = {}
        classes = results_df['y_true'].unique()

        for cls in classes:
            true_mask = results_df['y_true'] == cls #only ids for class of interest (DV=1 OR DV=0)
            correct_preds = (results_df['y_pred'][true_mask] == cls).sum() #number of correct predictions for class of interest
            total_true = true_mask.sum() #total true ids
            #recall = true positives / total actual positives
            recalls[cls] = correct_preds / total_true if total_true > 0 else 0.0

        #balanced accuracy = average of recalls for each class
        balanced_accuracy_manual = sum(recalls.values()) / len(recalls)
        f.write("\n\nManual Balanced Accuracy:\n")
        f.write(str(balanced_accuracy_manual))


#loading data
df = pd.read_csv("Cleaned_data_imputed.csv")

#adding target column
df['target'] = df['Dx'].apply(lambda x: 0 if x == 'HV' else 1)

#running functions (no '.csv' in output_file)
run_logistic_regression(df, "ARI_features_for_logreg.csv", "ARI_LogReg")
run_logistic_regression(df, "SCARED_features_for_logreg.csv", "SCARED_LogReg")
