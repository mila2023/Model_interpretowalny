import pandas as pd
import numpy as np
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("zbiór_10.csv")

X = data.drop(columns=["default"])
y = data["default"]

# dropujemy skurwysyna szczegolna forma wlasnosci
X = X.drop(columns="szczegolnaFormaPrawna_Symbol")

unique_values = X['formaWlasnosci_Symbol'].unique()
print(unique_values)

## potem jeszcze 'pkdKod'
categorical_cols = ['formaWlasnosci_Symbol']
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# lista kolumn OHE odpowiadających symbolom form własności
numeric_cols = [
    'ohe_fw_214','ohe_fw_215','ohe_fw_113','ohe_fw_216','ohe_fw_225','ohe_fw_226',
    'ohe_fw_224','ohe_fw_227','ohe_fw_234','ohe_fw_111','ohe_fw_112','ohe_fw_235',
    'ohe_fw_132','ohe_fw_123','ohe_fw_133','ohe_fw_122','ohe_fw_338', 'ohe_fw_000'
]

# inicjalizacja OneHotEncoder z ustalonymi kategoriami
ohe = OneHotEncoder(
    categories=[sorted([int(c.split('_')[-1]) for c in numeric_cols])],
    sparse_output=False,  # zmiana z sparse -> sparse_output
    drop=None
)

# dopasowanie i transformacja
ohe_array = ohe.fit_transform(X[['formaWlasnosci_Symbol']])

# utworzenie DataFrame z odpowiednimi nazwami kolumn
df_ohe = pd.DataFrame(ohe_array, columns=numeric_cols, index=X.index)

# połączenie z oryginalnym df
df = pd.concat([X, df_ohe], axis=1)