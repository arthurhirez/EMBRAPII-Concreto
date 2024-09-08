import pandas as pd
import numpy as np
from future.backports.http.cookiejar import lwp_cookie_str
from selenium.webdriver.support.expected_conditions import element_to_be_selected

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVR

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


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
    # Feature analysed: 'Fck'
    for feat, lim in remove_params.items():
        if verbose: print(
            f'{"Instâncias eliminadas devido a falta de representatividade (Fck = " + feat + " MPA):" :<90}',
            len(data_processed[data_processed['Fck'] == lim]))
        data_processed = data_processed[data_processed['Fck'] != lim]

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


