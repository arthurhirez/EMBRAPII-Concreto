import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

def eval_fulldata(X_train, X_test, y_tr, y_ts, classifiers):
    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both the training and test data
    X_tr = scaler.fit_transform(X_train)
    X_ts = scaler.transform(X_test)

    # Initialize a dictionary to store results
    results = {}

    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        # Train the model
        classifier.fit(X_tr, y_tr)

        # Make predictions
        y_pred = classifier.predict(X_ts)

        # Calculate performance metrics
        accuracy = accuracy_score(y_ts, y_pred)
        precision = precision_score(y_ts, y_pred, average = 'weighted', zero_division = 0)
        recall = recall_score(y_ts, y_pred, average = 'weighted')
        f1 = f1_score(y_ts, y_pred, average = 'weighted')

        # Calculate class distribution percentages
        class_distributions = y_tr.value_counts(normalize = True) * 100

        # Store results
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        results[name]['Class Distribution (%)'] = pd.DataFrame(class_distributions).round(2).to_dict()
    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    return results_df, y_pred


def eval_CV(X_tr, X_ts, y_tr, y_ts, classifiers, n_splits = 5):
    # Initialize K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

    # Initialize a dictionary to store results
    results = {name: {'Accuracy_tr': 0., 'Precision_tr': 0., 'Recall_tr': 0., 'F1 Score_tr': 0.,
                      'Accuracy_val': 0., 'Precision_val': 0., 'Recall_val': 0., 'F1 Score_val': 0.,
                      'Accuracy_test': 0., 'Precision_test': 0., 'Recall_test': 0., 'F1 Score_test': 0.} for name in
               classifiers}

    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        accuracy_scores_tr, precision_scores_tr, recall_scores_tr, f1_scores_tr = [], [], [], []
        accuracy_scores_val, precision_scores_val, recall_scores_val, f1_scores_val = [], [], [], []
        accuracy_scores_test, precision_scores_test, recall_scores_test, f1_scores_test = [], [], [], []
        class_distributions = []

        for train_index, val_index in skf.split(X_tr, y_tr):
            X_train, X_val = X_tr.iloc[train_index], X_tr.iloc[val_index]
            y_train, y_val = y_tr.iloc[train_index], y_tr.iloc[val_index]

            # Scale the data inside the CV loop
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Fit the model
            classifier.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred_train = classifier.predict(X_train_scaled)
            y_pred_val = classifier.predict(X_val_scaled)

            # Calculate performance metrics for training and validation
            accuracy_scores_tr.append(accuracy_score(y_train, y_pred_train))
            precision_scores_tr.append(precision_score(y_train, y_pred_train, average = 'weighted', zero_division = 0))
            recall_scores_tr.append(recall_score(y_train, y_pred_train, average = 'weighted'))
            f1_scores_tr.append(f1_score(y_train, y_pred_train, average = 'weighted'))

            accuracy_scores_val.append(accuracy_score(y_val, y_pred_val))
            precision_scores_val.append(precision_score(y_val, y_pred_val, average = 'weighted', zero_division = 0))
            recall_scores_val.append(recall_score(y_val, y_pred_val, average = 'weighted'))
            f1_scores_val.append(f1_score(y_val, y_pred_val, average = 'weighted'))

            # Evaluate on the full test set for each fold
            X_ts_scaled = scaler.transform(X_ts)
            y_pred_test = classifier.predict(X_ts_scaled)

            accuracy_scores_test.append(accuracy_score(y_ts, y_pred_test))
            precision_scores_test.append(precision_score(y_ts, y_pred_test, average = 'weighted', zero_division = 0))
            recall_scores_test.append(recall_score(y_ts, y_pred_test, average = 'weighted'))
            f1_scores_test.append(f1_score(y_ts, y_pred_test, average = 'weighted'))

            # Calculate class distribution percentages
            class_distribution = y_tr.value_counts(normalize = True) * 100
            class_distributions.append(dict(class_distribution))

        # Store mean results for CV (train, validation, and test)
        results[name]['Accuracy_tr'] = sum(accuracy_scores_tr) / len(accuracy_scores_tr)
        results[name]['Precision_tr'] = sum(precision_scores_tr) / len(precision_scores_tr)
        results[name]['Recall_tr'] = sum(recall_scores_tr) / len(recall_scores_tr)
        results[name]['F1 Score_tr'] = sum(f1_scores_tr) / len(f1_scores_tr)

        results[name]['Accuracy_val'] = sum(accuracy_scores_val) / len(accuracy_scores_val)
        results[name]['Precision_val'] = sum(precision_scores_val) / len(precision_scores_val)
        results[name]['Recall_val'] = sum(recall_scores_val) / len(recall_scores_val)
        results[name]['F1 Score_val'] = sum(f1_scores_val) / len(f1_scores_val)

        results[name]['Accuracy_test'] = sum(accuracy_scores_test) / len(accuracy_scores_test)
        results[name]['Precision_test'] = sum(precision_scores_test) / len(precision_scores_test)
        results[name]['Recall_test'] = sum(recall_scores_test) / len(recall_scores_test)
        results[name]['F1 Score_test'] = sum(f1_scores_test) / len(f1_scores_test)

        # results[name]['Sampling Method'] = sampling_method if sampling_method else 'None'
        results[name]['Class Distribution (%)'] = pd.DataFrame(class_distributions).mean().round(2).to_dict()

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    return results_df


def eval_classifier_CV(X_tr, X_ts, y_tr, y_ts, classifiers = None, CV = False, n_splits = 5):
    if classifiers is None:
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
            'Random Forest Classifier': RandomForestClassifier(random_state = 42)
            }

    return eval_CV(X_tr, X_ts, y_tr, y_ts, classifiers, n_splits = n_splits) if CV else \
        eval_fulldata(X_tr, X_ts, y_tr, y_ts, classifiers)






