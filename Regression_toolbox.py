from PreProcessing_toolbox import *

import pandas as pd
import numpy as np
import sklearn

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVR

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold



from scipy.stats import levene, bartlett
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def run_regression_CV(data, drop_columns, target, n_splits = 5):
    X = data.drop(drop_columns, axis = 1)
    y = data[target]

    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

    # Define the regression algorithms
    regressors = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor()
        }

    # Initialize a dictionary to store results
    results = {name: {'MAE': [], 'MSE': [], 'R2': []} for name in regressors}

    # Train and evaluate each regressor
    for name, regressor in regressors.items():
        mae_scores, mse_scores, r2_scores = [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Scale the data
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train the model
            regressor.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = regressor.predict(X_test_scaled)

            # Calculate performance metrics
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        # Store mean results
        results[name]['MAE'] = sum(mae_scores) / len(mae_scores)
        results[name]['MSE'] = sum(mse_scores) / len(mse_scores)
        results[name]['R2'] = sum(r2_scores) / len(r2_scores)

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    # Print the report
    print("Regression Algorithms Performance Report:")
    print(results_df)


def calculate_aic(n, mse, k):
    """
    Calculate AIC based on number of samples (n), mean squared error (mse),
    and number of parameters (k).
    """
    return n * np.log(mse) + 2 * k


def aic_scorer(estimator, X, y):
    """
    Custom AIC scorer to be used in GridSearchCV.
    """
    y_pred = estimator.predict(X)
    mse = mean_squared_error(y, y_pred)
    n = len(y)  # number of samples
    k = X.shape[1]  # number of features/parameters
    aic = calculate_aic(n, mse, k)
    return -aic  # Since GridSearchCV maximizes the score, return negative AIC


def grid_search_FS_RF(X, y, param_grid, verbose = True, n_splits = 5):
    # Define multiple metrics for regression
    scoring = {
        'r2': make_scorer(r2_score),
        'mse': make_scorer(mean_squared_error),
        'aic': aic_scorer  # Custom AIC scorer
        }

    # Initialize RandomForestRegressor
    rf = RandomForestRegressor(random_state = 42)

    # Set up KFold cross-validation
    kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

    # Set up GridSearchCV for RandomForestRegressor
    grid_search_rf = GridSearchCV(
        estimator = rf,  # Just RandomForestRegressor
        param_grid = param_grid,
        cv = kfold,
        scoring = scoring,  # Use multiple metrics including AIC
        refit = 'aic',  # The metric to use for the final model selection
        verbose = 1
        )

    # Fit the grid search for RandomForestRegressor
    grid_search_rf.fit(X, y)

    # Get the best RandomForestRegressor parameters
    best_rf = grid_search_rf.best_params_
    print("Best parameters for RandomForestRegressor: ", best_rf)

    # Get the best estimator (the one that gave the best result)
    best_model = grid_search_rf.best_estimator_

    # Get the feature importances from the best model
    feature_importances = best_model.feature_importances_

    # Create a DataFrame for better readability
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
        }).sort_values(by = 'Importance', ascending = False)

    if verbose:
        print("\nFeature Importances for the Best Model:")
        print(feature_importances_df)

    return grid_search_rf, feature_importances_df


def grid_search_RFClassifier(x_train, y_train, param_grid, n_splits = 5):
    # Define the hyperparameter grid for Random Forest

    scoring = {
        'r2': make_scorer(r2_score),
        'mse': make_scorer(mean_squared_error),
        'aic': aic_scorer  # Custom AIC scorer
        }

    # param_grid = {
    #     'n_estimators': [20, 50, 100, 200],
    #     'max_depth': [None, 3, 5, 10, 20],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'bootstrap': [True, False]
    # }

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(random_state = 42)

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)

    # Perform Grid Search with K-Fold Cross-Validation
    grid_search = GridSearchCV(estimator = rf,
                               param_grid = param_grid,
                               scoring = scoring,  # Use multiple metrics including AIC
                               refit = 'aic',  # The metric to use for the final model selection
                               cv = kf,
                               n_jobs = -1,
                               verbose = 2)

    # Fit the model on the training data
    grid_search.fit(x_train, y_train)

    # Get the best RandomForestRegressor parameters
    best_rf = grid_search.best_params_
    print("Best parameters for RandomForestRegressor: ", best_rf)

    return grid_search








def eval_regressor(X_tr, X_ts, y_tr, y_ts, regressors = None):
    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both the training and test data
    X_train_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_ts)

    # Initialize a dictionary to store results
    results = {}

    if type(regressors) != dict:
        regressors = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor()
            }

    # Train and evaluate each regressor
    for name, regressor in regressors.items():
        # Train the model
        regressor.fit(X_train_scaled, y_tr)

        # Make predictions
        y_pred = regressor.predict(X_test_scaled)

        # Calculate performance metrics
        mae = mean_absolute_error(y_ts, y_pred)
        mse = mean_squared_error(y_ts, y_pred)
        r2 = r2_score(y_ts, y_pred)

        # Store results
        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    # Print the report
    print("Regression Algorithms Performance Report:")
    print(results_df)


def eval_regressor_CV(X_tr, X_ts, y_tr, y_ts, regressors = None, CV = False, n_splits = 5):
    if type(regressors) != dict:
        regressors = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor()
            }

    return eval_CV(X_tr, X_ts, y_tr, y_ts, regressors, n_splits = n_splits) if CV else \
    eval_fulldata(X_tr, X_ts, y_tr, y_ts, regressors)



def eval_fulldata(X_tr, X_ts, y_tr, y_ts, regressors):
    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform both the training and test data
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)

    # Initialize a dictionary to store results
    results = {}

    # Train and evaluate each regressor
    for name, regressor in regressors.items():
        # Train the model
        regressor.fit(X_tr, y_tr)

        # Make predictions
        y_pred = regressor.predict(X_ts)

        # Calculate performance metrics
        mae = mean_absolute_error(y_ts, y_pred)
        mse = mean_squared_error(y_ts, y_pred)
        r2 = r2_score(y_ts, y_pred)

        # Store results
        results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    # Print the report
    # print("Regression Algorithms Performance Report:")
    # print(results_df)

    return results_df

def fit_regression(X_tr, X_ts, y_tr, regressor):
    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Scale the data
    X_train_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_ts)

    # Train the model
    regressor.fit(X_train_scaled, y_tr)

    # Make predictions
    y_pred = regressor.predict(X_test_scaled)

    return regressor, y_pred




def eval_CV(X_tr, X_ts, y_tr, y_ts, regressors, n_splits=5):
    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize a dictionary to store results
    results = {name: {'MAE_tr': 0., 'MSE_tr': 0., 'R2_tr': 0.,
                      'MAE_val': 0., 'MSE_val': 0., 'R2_val': 0.,
                      'MAE_test': 0., 'MSE_test': 0., 'R2_test': 0.} for name in regressors}

    # Train and evaluate each regressor
    for name, regressor in regressors.items():
        mae_scores_tr, mse_scores_tr, r2_scores_tr = [], [], []
        mae_scores_val, mse_scores_val, r2_scores_val = [], [], []
        mae_scores_test, mse_scores_test, r2_scores_test = [], [], []

        for train_index, val_index in kf.split(X_tr):
            X_train, X_val = X_tr.iloc[train_index], X_tr.iloc[val_index]
            y_train, y_val = y_tr.iloc[train_index], y_tr.iloc[val_index]

            # Scale the data inside the CV loop
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Fit the model
            regressor.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred_train = regressor.predict(X_train_scaled)
            y_pred_val = regressor.predict(X_val_scaled)

            # Calculate performance metrics for training and validation
            mae_scores_tr.append(mean_absolute_error(y_train, y_pred_train))
            mse_scores_tr.append(mean_squared_error(y_train, y_pred_train))
            r2_scores_tr.append(r2_score(y_train, y_pred_train))

            mae_scores_val.append(mean_absolute_error(y_val, y_pred_val))
            mse_scores_val.append(mean_squared_error(y_val, y_pred_val))
            r2_scores_val.append(r2_score(y_val, y_pred_val))

            # Evaluate on the full test set for each fold
            X_ts_scaled = scaler.transform(X_ts)
            y_pred_test = regressor.predict(X_ts_scaled)

            mae_scores_test.append(mean_absolute_error(y_ts, y_pred_test))
            mse_scores_test.append(mean_squared_error(y_ts, y_pred_test))
            r2_scores_test.append(r2_score(y_ts, y_pred_test))

        # Store mean results for CV (train, validation, and test)
        results[name]['MAE_tr'] = sum(mae_scores_tr) / len(mae_scores_tr)
        results[name]['MSE_tr'] = sum(mse_scores_tr) / len(mse_scores_tr)
        results[name]['R2_tr'] = sum(r2_scores_tr) / len(r2_scores_tr)

        results[name]['MAE_val'] = sum(mae_scores_val) / len(mae_scores_val)
        results[name]['MSE_val'] = sum(mse_scores_val) / len(mse_scores_val)
        results[name]['R2_val'] = sum(r2_scores_val) / len(r2_scores_val)

        results[name]['MAE_test'] = sum(mae_scores_test) / len(mae_scores_test)
        results[name]['MSE_test'] = sum(mse_scores_test) / len(mse_scores_test)
        results[name]['R2_test'] = sum(r2_scores_test) / len(r2_scores_test)

    # Convert results to DataFrame for better readability
    results_df = pd.DataFrame(results).T

    return results_df



# def eval_CV(X_tr, X_ts, y_tr, y_ts, regressors, n_splits = 5):
#     # Initialize the Min-Max Scaler
#     scaler = MinMaxScaler()
#     X_tr_scaled = scaler.fit_transform(X_tr)
#     X_ts_scaled = scaler.transform(X_ts)
#
#     # Initialize K-Fold Cross-Validation
#     kf = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
#
#     # Initialize a dictionary to store results
#     results = {name: {'MAE_tr': 0., 'MSE_tr': 0., 'R2_tr': 0.,
#                       'MAE_val': 0., 'MSE_val': 0., 'R2_val': 0.} for name in regressors}
#
#     # Train and evaluate each regressor
#     for name, regressor in regressors.items():
#         # Train the model w/ full dataset
#         regressor.fit(X_tr_scaled, y_tr)
#
#         # Make predictions
#         y_pred_full = regressor.predict(X_ts_scaled)
#
#         # Calculate performance metrics
#         results[name]['MAE'] = mean_absolute_error(y_ts, y_pred_full)
#         results[name]['MSE'] = mean_squared_error(y_ts, y_pred_full)
#         results[name]['R2'] = r2_score(y_ts, y_pred_full)
#
#
#         mae_scores, mse_scores, r2_scores = [], [], []
#
#         for train_index, test_index in kf.split(X_tr):
#             X_train, X_test = X_tr.iloc[train_index], X_tr.iloc[test_index]
#             y_train, y_test = y_tr.iloc[train_index], y_tr.iloc[test_index]
#
#             # Make predictions
#             _, y_pred = fit_regression(X_train, X_test, y_train, regressor)
#
#             # Calculate performance metrics
#             mae_scores.append(mean_absolute_error(y_test, y_pred))
#             mse_scores.append(mean_squared_error(y_test, y_pred))
#             r2_scores.append(r2_score(y_test, y_pred))
#
#         # Store mean results
#         results[name]['MAE_CV'] = sum(mae_scores) / len(mae_scores)
#         results[name]['MSE_CV'] = sum(mse_scores) / len(mse_scores)
#         results[name]['R2_CV'] = sum(r2_scores) / len(r2_scores)
#
#
#
#     # Convert results to DataFrame for better readability
#     results_df = pd.DataFrame(results).T
#
#     # Print the report
#     # print("Regression Algorithms Performance Report:")
#     # print(results_df)
#
#     return results_df

def run_experiment_ciment_fck(X_tr, X_ts, y_tr, y_ts, param_grid_feat_selection = None, param_grid_regressor = None,
                              n_splits = 5, group_columns = ['cimento Tipo', 'cimento Classe de resistência', 'Fck']):
    if not isinstance(param_grid_regressor, dict):
        param_grid_feat_selection = {
            'n_estimators': [25, 75, 150],
            'max_depth': [2, 10],
            }

    if not isinstance(param_grid_regressor, dict):
        param_grid_regressor = {
            'n_estimators': [25, 75, 150],
            'max_depth': [3, 15],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
            }

    ciment_types = ['CP II', ['CPIII', 'CP IV'], 'CPV']

    results = {}

    for ciment in ciment_types:
        if isinstance(ciment, list):
            # If ciment is a list, join the list elements with a regex 'OR' (|) for contains
            pattern = '|'.join(ciment)
        else:
            # If ciment is a string, use it directly
            pattern = ciment

        results[pattern] = {}

        print(15 * "**-----**")

        tr_CP = pd.concat([X_tr, y_tr], axis = 1)
        ts_CP = pd.concat([X_ts, y_ts], axis = 1)

        tr_CP = tr_CP[tr_CP['cimento Tipo'].str.contains(pattern)]
        ts_CP = ts_CP[ts_CP['cimento Tipo'].str.contains(pattern)]

        print(f'\n\nCiment: {pattern}\t "Treino": {tr_CP.shape}\t "Teste": {ts_CP.shape}')

        X_tr_CP = tr_CP.drop(columns = 'Fc 28d')
        y_tr_CP = tr_CP['Fc 28d']

        X_ts_CP = ts_CP.drop(columns = 'Fc 28d')
        y_ts_CP = ts_CP['Fc 28d']

        for fck in tr_CP['Fck'].sort_values().unique():

            tr_CP_fck = tr_CP[tr_CP['Fck'] == fck]

            if len(tr_CP_fck[tr_CP_fck['cimento Tipo'].str.contains(pattern)]) == 0:
                pass
            else:
                tr_CP_fck_PP = preprocess_data(data = tr_CP_fck,
                                               params_filter = {},
                                               verbose = False)

                X_tr_CP_fck = tr_CP_fck_PP.drop(columns = 'Fc 28d')
                y_tr_CP_fck = tr_CP_fck_PP['Fc 28d']

                ts_CP_fck = ts_CP[ts_CP['Fck'] == fck]
                ts_CP_fck_PP = ts_CP_fck[ts_CP_fck['cimento Tipo'].str.contains(pattern)]
                ts_CP_fck_PP = ts_CP_fck_PP[tr_CP_fck_PP.columns]

                X_ts_CP_fck = ts_CP_fck_PP.drop(columns = 'Fc 28d')
                y_ts_CP_fck = ts_CP_fck_PP['Fc 28d']

                print(f'\n\nFck: {fck}\t Treino: {X_tr_CP_fck.shape}\t Validação: {X_ts_CP_fck.shape}')

                print('\nGrid search - Feature selection')
                feature_selector_fck = grid_search_FS_RF(X = X_tr_CP_fck.drop(columns = group_columns),
                                                         y = y_tr_CP_fck,
                                                         param_grid = param_grid_feat_selection,
                                                         n_splits = n_splits,
                                                         verbose = False)

                feat_importance_fck = feature_selector_fck[1]
                features_selected_fck = feat_importance_fck['Feature'].tolist()[:15]
                X_tr_CP_FS_fck = X_tr_CP_fck[features_selected_fck]
                X_ts_CP_FS_fck = X_ts_CP_fck[features_selected_fck]

                print('\nGrid search - Regressor')
                rf_gridsearch_CP_fck = grid_search_RFClassifier(x_train = X_tr_CP_FS_fck,
                                                                y_train = y_tr_CP_fck,
                                                                param_grid = param_grid_regressor,
                                                                n_splits = n_splits)

                results[pattern][fck] = eval_regressor_CV(X_tr_CP_FS_fck,
                                                          X_ts_CP_FS_fck,
                                                          y_tr_CP_fck,
                                                          y_ts_CP_fck,
                                                          regressors = {
                                                              'Random Forest Regressor': rf_gridsearch_CP_fck.best_estimator_},
                                                          CV = True,
                                                          n_splits = n_splits)

                results[pattern][fck]['Shape_treino'] = str(X_tr_CP_FS_fck.shape)
                results[pattern][fck]['Shape_teste'] = str(X_ts_CP_FS_fck.shape)

                display(results[pattern][fck][
                            ['Shape_treino', 'Shape_teste', 'MAE_tr', 'R2_tr', 'MAE_val', 'R2_val', 'MAE_test',
                             'R2_test']])

        print(f'\n\nCiment: {pattern}\t Treino: {X_tr_CP.shape}\t Validação: {y_tr_CP.shape}')
        
        print('\nGrid search - Feature selection')
        feature_selector = grid_search_FS_RF(X = X_tr_CP.drop(columns = group_columns),
                                             y = y_tr_CP,
                                             param_grid = param_grid_feat_selection,
                                             n_splits = n_splits,
                                             verbose = False)

        feat_importance = feature_selector[1]
        features_selected = feat_importance['Feature'].tolist()[:15]

        X_tr_CP_FS = X_tr_CP[features_selected]
        X_ts_CP_FS = X_ts_CP[features_selected]

        print('\nGrid search - Regressor')
        rf_gridsearch_CP = grid_search_RFClassifier(x_train = X_tr_CP_FS,
                                                    y_train = y_tr_CP,
                                                    param_grid = param_grid_regressor,
                                                    n_splits = 3)

        results[pattern]['all'] = eval_regressor_CV(X_tr_CP_FS, X_ts_CP_FS, y_tr_CP, y_ts_CP,
                                                    regressors = {
                                                        'Random Forest Regressor': rf_gridsearch_CP.best_estimator_},
                                                    CV = True,
                                                    n_splits = 3)

        results[pattern]['all']['Shape_treino'] = str(X_tr_CP_FS.shape)
        results[pattern]['all']['Shape_teste'] = str(X_ts_CP_FS.shape)

        display(results[pattern]['all'])

        final_df = get_df_results(results)

        print('\n Compilado resultados:')
        display(final_df[final_df['Model'] == 'Random Forest Regressor'][
                    ['CP_Fck', 'Model', 'MAE_tr', 'R2_tr', 'MAE_test', 'R2_test', 'Shape_treino', 'Shape_teste']])

        plot_residuals(X_tr_CP[features_selected + ['Fck']], X_ts_CP[features_selected + ['Fck']], y_tr_CP, y_ts_CP,
                       group_columns, regressors = rf_gridsearch_CP)
        plot_residuals_with_rolling_std(X_tr_CP[features_selected + ['Fck']], X_ts_CP[features_selected + ['Fck']],
                                        y_tr_CP, y_ts_CP, group_columns, window_size = 30)

def get_df_results(dict_results):
    # Initialize an empty list to store the dataframes
    df_list = []

    # Iterate through the dictionary
    for CP, data_fck in dict_results.items():
        for ciment, data in data_fck.items():
            # Add a column 'Condition' with the current condition value
            df = data.copy()
            df['CP_Fck'] = str(CP) + '_' + str(ciment)
            # Append the dataframe to the list
            df_list.append(df)

    # Concatenate all the dataframes into one
    final_df = pd.concat(df_list)

    # Reset the index for cleanliness
    final_df.reset_index(inplace = True, names = 'Model')

    # Reorder the columns to place 'Condition' first
    columns = ['CP_Fck'] + [col for col in final_df.columns if col != 'CP_Fck']
    return final_df[columns]



def plot_residuals(x_train, x_test, y_train, y_test, group_columns, regressors = None, verbose = False):

    rf = regressors.best_estimator_ \
        if (type(regressors) == sklearn.model_selection._search.GridSearchCV) \
        else RandomForestRegressor(random_state = 42)
    rf.fit(x_train, y_train)

    # Make predictions
    y_pred = rf.predict(x_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # General Performance Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

    # Get unique 'Fck' values and assign colors
    unique_fck = x_test['Fck'].sort_values().unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_fck)))  # Use a colormap to get colors

    # Create the figure and axes objects
    fig, axs = plt.subplots(1, 2, figsize = (12, 6))

    # First plot: KDE plot for all residuals (no hue)
    sns.kdeplot(residuals, fill = False, color = 'gray', linestyle = '--', lw = 3, alpha = 0.7, ax = axs[0])
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Density')
    axs[0].set_title('KDE Plot of Residuals (No Hue)')

    # Second plot: KDE plot of residuals, hued by 'Fck'
    for i, fck_value in enumerate(unique_fck):
        sns.kdeplot(
            residuals[x_test['Fck'] == fck_value],
            fill = False, label = f'Fck = {fck_value}', color = colors[i], alpha = 0.6, ax = axs[0]
            )

    axs[0].axvline(0, color = 'black', linestyle = 'dotted')
    axs[0].set_xlabel('Residuals')
    axs[0].set_ylabel('Density')
    axs[0].set_title('KDE Plot of Residuals (Hue by Fck)')
    axs[0].legend(title = 'Fck Values')

    # Residual plot (2nd plot) with hue by 'Fck'
    for i, fck_value in enumerate(unique_fck):
        mask = x_test['Fck'] == fck_value
        axs[1].scatter(y_pred[mask], residuals[mask], color = colors[i], label = f'Fck = {fck_value}', alpha = 0.4)

    axs[1].axhline(0, color = 'black', linestyle = 'dotted')
    axs[1].axhline(3 * np.std(residuals), color = 'black', linestyle = 'dotted')
    axs[1].axhline(-3 * np.std(residuals), color = 'black', linestyle = 'dotted')

    axs[1].set_xlabel('Predicted Values')
    axs[1].set_ylabel('Residuals')
    axs[1].set_title('Residual Plot (Hue by Fck)')
    axs[1].legend(title = 'Fck Values')

    plt.tight_layout()
    plt.show()

    if verbose:
        # Check for potential outliers
        outliers = np.where(np.abs(residuals) > 3 * np.std(residuals))[0]
        print(f'Number of outliers: {len(outliers)}')

        print(x_test.iloc[outliers].groupby(group_columns).count()['CT Cimento'])


def plot_residuals_with_rolling_std(x_train, x_test, y_train, y_test, group_columns, window_size = 50,
                                    regressors = None, verbose = False):

    # Use the best estimator from the grid search
    rf = regressors.best_estimator_ \
        if (type(regressors) == sklearn.model_selection._search.GridSearchCV) \
        else RandomForestRegressor(random_state = 42)

    rf.fit(x_train, y_train)

    # Make predictions and calculate residuals
    y_pred = rf.predict(x_test)
    residuals = y_test - y_pred

    # Calculate standard deviation of residuals for variance bands
    residual_std = np.std(residuals)

    # Get unique 'Fck' values and assign colors
    unique_fck = x_test['Fck'].sort_values().unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_fck)))

    # Create subplots for each 'Fck' class
    fig, axs = plt.subplots(1, len(unique_fck), figsize = (15, 6), sharey = True)

    # Loop through each unique 'Fck' and plot residuals
    for i, fck_value in enumerate(unique_fck):
        mask = x_test['Fck'] == fck_value

        # Scatter plot of residuals for each Fck class
        sns.scatterplot(x = y_pred[mask], y = residuals[mask], color = colors[i], label = f'Fck = {fck_value}',
                        alpha = 0.4, ax = axs[i])

        # Regression line with confidence interval
        sns.regplot(x = y_pred[mask], y = residuals[mask], scatter = False, robust = True, color = 'blue',
                    label = 'Linear Regression',
                    line_kws = {'lw': 3}, ci = None, ax = axs[i])

        # Horizontal lines for 0 and ±3 std deviations
        axs[i].axhline(0, color = 'black', linestyle = 'dotted')
        axs[i].axhline(3 * residual_std, color = 'black', linestyle = 'dotted')
        axs[i].axhline(-3 * residual_std, color = 'black', linestyle = 'dotted')

        # Sort by predicted values and calculate rolling standard deviation
        sorted_indices = np.argsort(y_pred[mask])
        sorted_y_pred = y_pred[mask][sorted_indices]
        sorted_residuals = residuals[mask].iloc[sorted_indices]  # Use .iloc to properly index

        # Rolling standard deviation
        rolling_std = pd.Series(sorted_residuals).rolling(window = window_size).std()

        # Plot rolling standard deviation
        axs[i].plot(sorted_y_pred, rolling_std, color = 'red', label = 'Rolling Std Dev')

        # Formatting
        axs[i].set_xlabel('Predicted Values')
        axs[i].set_ylabel('Residuals')
        axs[i].set_title(f'Residual Plot for Fck={fck_value}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    if verbose:
        # Check for potential outliers
        outliers = np.where(np.abs(residuals) > 3 * np.std(residuals))[0]
        print(f'Number of outliers: {len(outliers)}')

        print(x_test.iloc[outliers].groupby(group_columns).count()['CT Cimento'])


def plot_residuals_binned_variance(x_train, x_test, y_train, y_test, n_bins = 10, regressor = None):
    # Use the best estimator from the grid search
    rf = regressor.best_estimator_ \
        if (type(regressor) == sklearn.model_selection._search.GridSearchCV) \
        else RandomForestRegressor(random_state = 42)

    rf.fit(x_train, y_train)

    # Make predictions
    y_pred = rf.predict(x_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Bin the predicted values and calculate variance within each bin
    bins = np.linspace(min(y_pred), max(y_pred), n_bins + 1)  # +1 to include the upper edge of the last bin
    bin_indices = np.digitize(y_pred, bins)
    binned_variance = [np.var(residuals[bin_indices == i]) for i in range(1, len(bins))]

    # Calculate bin midpoints for x-axis labels
    bin_midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Plot variance per bin
    plt.figure(figsize = (10, 6))
    plt.bar(bin_midpoints, binned_variance, width = np.diff(bins), color = 'blue', alpha = 0.7, edgecolor = 'black')
    plt.xlabel('Predicted Values')
    plt.ylabel('Variance of Residuals')
    plt.title('Variance of Residuals Binned by Predicted Values')
    plt.xticks(bin_midpoints, [f'{v:.2f}' for v in bin_midpoints], rotation = 45)  # Set x-ticks to bin midpoints
    plt.tight_layout()
    plt.show()





def test_variance_homogeneity(x_train, x_test, y_train, y_test, n_bins = 5, regressor = None):
    # Use the best estimator from the grid search
    rf = regressor.best_estimator_ \
        if (type(regressor) == sklearn.model_selection._search.GridSearchCV) \
        else RandomForestRegressor(random_state = 42)

    rf.fit(x_train, y_train)

    # Make predictions
    y_pred = rf.predict(x_test)

    # Calculate residuals
    residuals = y_test - y_pred

    # Bin the predicted values
    bins = np.linspace(min(y_pred), max(y_pred), n_bins)
    bin_indices = np.digitize(y_pred, bins)
    binned_residuals = [residuals[bin_indices == i] for i in range(1, len(bins))]

    # Perform Levene's test and Bartlett's test
    levene_test = levene(*binned_residuals)
    bartlett_test = bartlett(*binned_residuals)

    print(f"Levene's Test: Statistic = {levene_test.statistic}, p-value = {levene_test.pvalue}")
    print(f"Bartlett's Test: Statistic = {bartlett_test.statistic}, p-value = {bartlett_test.pvalue}")