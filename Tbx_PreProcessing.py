import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, KFold


def preprocess_data(data, params_filter, verbose = True):
    data_processed = data.copy()
    data_processed = data_processed[data_processed['CT Água'] != 0]

    data_processed['AC'] = data_processed['CT Água'] / data_processed['CT Cimento']

    # try:
    #     data_processed = data_processed.drop(columns = ['Planta', 'Empresa'])  # remove "empresa" and "planta"
    # except:
    #     data_processed = data_processed

    for c in data_processed:
        try:
            data_processed[c] = data_processed[c].astype(float)
        except:
            # data_processed[c] = data_processed[c]
            if verbose: print(f'Feature {c} não é numerica\n')

    for feat, lim in params_filter.items():
        data_processed = data_processed[data_processed[feat] < lim]

    columns_all_zero = data_processed.columns[(data_processed == 0).all()]
    data_processed = data_processed.drop(columns = columns_all_zero)

    data_processed = data_processed.drop_duplicates()

    instances_per_class = data_processed.groupby(['cimento Tipo', 'cimento Classe de resistência']).count()[
        'Fck'].reset_index()

    keep_instances = instances_per_class[instances_per_class['Fck'] > 10].reset_index()

    data_processed = data_processed[data_processed['cimento Tipo'].isin(keep_instances['cimento Tipo'])]

    data_processed['Brita_total'] = data_processed['CT Brita 0'] + data_processed['CT Brita 1']
    data_processed['Areia_total'] = data_processed['CT Areia natural'] + data_processed['CT Areia artificial']
    data_processed['Agregados'] = data_processed['Brita_total'] + data_processed['Areia_total']

    for c in ['CT Brita 0', 'CT Brita 1', 'CT Areia natural', 'CT Areia artificial', 'Brita_total', 'Areia_total',
              'Agregados']:
        data_processed[c + '_Cimento'] = data_processed[c] / data_processed['CT Cimento']

    return data_processed


def apply_standardization_filter(data, remove_params, verbose = True, regression = True):
    # TODO: implement filter to verify class/fck_min
    # concrete_class = 'CP V'
    # age = '7_days'

    data_processed = data.copy()  # Caso precise incluir features

    # Feature analysed: 'Fck'
    for feat, lim in remove_params.items():
        if verbose: print(
            f'{"Instâncias eliminadas devido a falta de representatividade (Fck = " + feat + " MPA):" :<90}',
            len(data_processed[data_processed['Fck'] == lim]))
        data_processed = data_processed[data_processed['Fck'] != lim]

    # Feature analysed: 'AC'
    if verbose: print(f'{"Instâncias eliminadas devido não conformidade NBR6118:2023 (relação água/cimento):" :<90}',
                      len(data_processed[data_processed['AC'] > 0.65]))
    data_processed = data_processed[data_processed['AC'] <= 0.65]

    if regression:
        # Feature analysed: 'Fc 28d'
        ## Precisa de análise mais detalhada!!!
        data_processed['Status'] = np.where(data_processed['Fc 28d'] >= data_processed['Fck'], 1, 0)
        if verbose: print(f'{"Instâncias eliminadas devido não conformidade resistência especificada:" : <90}',
                          len(data_processed[data_processed['Status'] == 0]))
        data_processed = data_processed[data_processed['Status'] == 1]

        data_processed.drop(columns = ['Status'], inplace = True)

    return data_processed

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



