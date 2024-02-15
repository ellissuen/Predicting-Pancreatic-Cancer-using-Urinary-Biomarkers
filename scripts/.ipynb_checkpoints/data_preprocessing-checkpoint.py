# ***Step 1. Deal with missing values***

# import modules for missing value imputation
from sklearn.impute import SimpleImputer, KNNImputer

# only columns 'plasma_CA19_9', and 'REG1A' have missing values
missing_value_columns = ['plasma_CA19_9', 'REG1A']

df_copy = df.copy()

# remove missing columns
def remove_missing():
    df_copy = df.drop(missing_value_columns, axis=1, inplace=True)
    return df_copy

# replace missing values with mean
def mean_missing():
    for column in missing_value_columns:
        df_copy[column].fillna(df[column].mean(), inplace=True)
    return df_copy
    

# replace missing values using KNN
def knn_missing:
    imputer = KNNImputer()
    df_copy = imputer.fit_transform(df)
    return df_filled
    
    
# ***Step 2. Scaling / Normalization techniques***

# ***Step 3. Feature selection***
