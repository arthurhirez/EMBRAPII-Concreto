from sklearn.model_selection import StratifiedKFold

from Tbx_PreProcessing import *


def grid_search_FS_RF(X, y, param_grid, verbose=True, n_splits=5, regression=True, refit_metric=None):
    # Define default metrics for regression and classification
    default_regr_metric = 'aic'
    default_clas_metric = 'f1'

    # Define multiple metrics for regression and classification
    if regression:
        scoring = {
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error),
            'aic': aic_scorer  # Custom AIC scorer
        }
        model = RandomForestRegressor(random_state=42)

        # Set default refit metric for regression if not provided or incorrect
        if refit_metric not in scoring:
            print(f"Warning: '{refit_metric}' is not valid for regression. Defaulting to '{default_regr_metric}'.")
            refit_metric = default_regr_metric
    else:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            # 'roc_auc': make_scorer(roc_auc_score)  # Use AUC for classifier
        }
        model = RandomForestClassifier(random_state=42)

        # Set default refit metric for classification if not provided or incorrect
        if refit_metric not in scoring:
            print(f"Warning: '{refit_metric}' is not valid for classification. Defaulting to '{default_clas_metric}'.")
            refit_metric = default_clas_metric

    # Set up KFold cross-validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring=scoring,
        refit=refit_metric,  # Metric used for final model selection
        verbose=1
    )

    # Fit the grid search for the model
    grid_search.fit(X, y)

    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    print(f"Best parameters for {'RandomForestRegressor' if regression else 'RandomForestClassifier'}: ", best_params)

    best_model = grid_search.best_estimator_

    # Get feature importances (only applicable for tree-based models)
    feature_importances = best_model.feature_importances_

    # Create a DataFrame for better readability
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    if verbose:
        print("\nFeature Importances for the Best Model:")
        print(feature_importances_df)

    return grid_search, feature_importances_df



def grid_search_RF(x_train, y_train, param_grid, regression = False, refit_metric = 'f1', n_splits = 5):


    # Define scoring metrics
    if regression:
        scoring = {
            'r2': make_scorer(r2_score),
            'mse': make_scorer(mean_squared_error),
            'aic': aic_scorer  # Custom AIC scorer
            }

        valid_metrics = ['aic', 'mse', 'r2']
        default_metric = 'mse'
        refit_metric = refit_metric if refit_metric in valid_metrics else default_metric
        estimator = RandomForestRegressor(random_state = 42)
        cv = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    else:
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average = 'weighted', zero_division = 0),
            'recall': make_scorer(recall_score, average = 'weighted'),
            'f1': make_scorer(f1_score, average = 'weighted')
            }
        valid_metrics = ['accuracy', 'precision', 'recall', 'f1']
        default_metric = 'f1'
        refit_metric = refit_metric if refit_metric in valid_metrics else default_metric
        estimator = RandomForestClassifier(random_state = 42)
        cv = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator = estimator,
                               param_grid = param_grid,
                               scoring = scoring,
                               refit = refit_metric,
                               cv = cv,
                               n_jobs = -1,
                               verbose = 2)

    # Fit the model on the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters for {'RandomForestRegressor' if regression else 'RandomForestClassifier'}: ", best_params)

    return grid_search