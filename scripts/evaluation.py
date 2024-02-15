# Step 6. Classification Model Reporting

# import necessary modules
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def outcome_reporting(X_test, y_test, clf, best_params, missing_desc, scaling_desc, feature_desc, selected_features, split_desc, model_desc):
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate the ROC curve and AUC
    y_probs = clf.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1
    label_encoder = LabelEncoder()
    y_test_numerical = label_encoder.fit_transform(y_test)

    fpr, tpr, thresholds = roc_curve(y_test_numerical, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Print steps taken
    print(f'Missing Value Imputation: {missing_desc}')
    print(f'Scaling/Normalization: {scaling_desc}')
    print(f'Feature Selection: {feature_desc}: {selected_features}')
    print(f'Data Split: {split_desc}')
    print(f'Classification Model: {model_desc}')
    
    # Print best parameters
    print(f'Best Parameters: {best_params}')
    
    # Print evaluation
    print(f'Accuracy: {accuracy}')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # Print ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    print("\n" + "-"*50 + "\n")  # Add a separator between dataframes
