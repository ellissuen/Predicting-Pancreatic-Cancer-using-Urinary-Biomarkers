# Step 5. Model selection

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
   
# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression,
    'Random Forest': RandomForestClassifier,
    'Gradient Boosting': GradientBoostingClassifier,
    'Support Vector Machine': SVC,
    'K-Nearest Neighbors': KNeighborsClassifier
}

# Define hyperparameters grid for each model
param_grids = {
    'Logistic Regression': {'max_iter': [100, 500, 1000]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'Support Vector Machine': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]}
}

# Define models and their corresponding hyperparameters grid
models_params = {
    classifiers[name]: (param_grid, name) for name, param_grid in param_grids.items()
}
