# import df from open_file.py
from open_file import open_file
df = open_file()

# import all necessary module for data preprocessing
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RepeatedKFold

########################################################################

# Change y ('diagnosis') from 1 = no cancer, 2 = benign, 3 = cancer to boolean
def classify_diagnosis(value):
    if value in [1, 2]:
        return '0'
    else:
        return '1'
    df['diagnosis'] = df['diagnosis'].apply(classify_diagnosis)

######################################################################## 

### Step 1. Deal with missing values

# only columns 'plasma_CA19_9', and 'REG1A' have missing values
missing_value_columns = ['plasma_CA19_9', 'REG1A']

# copy of df to change and return
df_copy = df.copy()

# remove missing columns
def remove_missing(data):
    df_copy = df.drop(missing_value_columns, axis=1)
    return df_copy

# replace missing values with mean
def mean_missing(data):
    df_copy = data.copy()
    for column in missing_value_columns:
        df_copy[column].fillna(df[column].mean(), inplace=True)
    return df_copy

# replace missing values using KNN
def knn_missing(data):
    imputer = KNNImputer()
    df_filled = imputer.fit_transform(df)
    df_copy = pd.DataFrame(df_filled, columns=df.columns)
    return df_copy

missing_value_functions = {
    remove_missing: "Remove Missing Values",
    mean_missing: "Replace with Mean",
    knn_missing: "Replace with KNN"
}

######################################################################## 

### Step 2. Scaling / Normalization techniques

# Using min max scaler
def min_max_scaler(data):
    scaler = MinMaxScaler()
    transformed_data = scaler.fit_transform(data)
    return transformed_data

# Using robust scaler
def robust_scaler(data):
    scaler = RobustScaler()
    transformed_data = scaler.fit_transform(data)
    return transformed_data

# Using quantile transformer
def quant_transformation(data):
    transformer = QuantileTransformer(n_quantiles=min(data.shape[0], 1000))
    transformed_data = transformer.fit_transform(data)
    return transformed_data

# Using log transformer
def log_transformation(data):
    transformed_data = np.log1p(data)  # Add 1 to avoid taking the log of zero
    return transformed_data

scaling_normalization_functions = {
    min_max_scaler: "Min-Max Scaling",
    robust_scaler: "Robust Scaling",
    quant_transformation: "Quantile Transformation",
    log_transformation: "Log Transformation"
}

########################################################################

### Step 3. Feature selection

# Using Recursive Feature Elimination
def rfe_feature(X, y, n_features_to_select=5):
    estimator = Lasso()
    selector = RFE(estimator, n_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    return selected_features

# Using Lasso Regression
def lasso_feature(X, y, alpha=0.0001):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    selected_features = X.columns[lasso.coef_ != 0]
    return selected_features

# Using Principal Component Analysis
def pca_feature(X, n_components):
    selected_features = PCA(n_components=n_components)
    selected_features.fit(X)
    return selected_features

feature_selection_functions = {
    rfe_feature: "Recursive Feature Elimination",
    lasso_feature: "Lasso Regularizer",
    pca_feature: "Principal Component Analysis"
}

########################################################################

### Step 4. Data split

# Using Train Test Split     
def train_test_datasplit(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

    
# Using Repeated K Fold     
def repeated_k_fold_datasplit(X, y):
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    splits = rkf.split(X, y)
    return splits

data_splitting_functions = {
    train_test_datasplit: "Train Test Split", 
    repeated_k_fold_datasplit: "Repeated K Fold"
}
