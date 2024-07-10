from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import calculate_custom_accuracy

def hyperparameter_tuning(tfidf_matrix, labels):
    """
    Perform hyperparameter tuning for RandomForestClassifier.

    Args:
        tfidf_matrix (sparse matrix): TF-IDF feature matrix.
        labels (list): List of labels.

    Returns:
        RandomForestClassifier: Best estimator from GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(tfidf_matrix, labels)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    return grid_search.best_estimator_

def train_and_evaluate_classifier(tfidf_matrix, labels, tfidf_vectorizer):
    """
    Train and evaluate the RandomForest classifier.

    Args:
        tfidf_matrix (sparse matrix): TF-IDF feature matrix.
        labels (list): List of labels.
        tfidf_vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.

    Returns:
        tuple: Trained classifier and fitted TF-IDF vectorizer.
    """
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    rf_classifier = hyperparameter_tuning(X_train_res, y_train_res)

    # Training with best parameters
    rf_classifier.fit(X_train_res, y_train_res)
    y_pred = rf_classifier.predict(X_test)

    # Evaluation
    print("Regular Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Calculate and print custom accuracy
    custom_acc = calculate_custom_accuracy(y_test, y_pred)
    print(f"Custom Accuracy (considering distances): {custom_acc:.2f}")

    # Cross-validation
    scores = cross_val_score(rf_classifier, tfidf_matrix, labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

    # Plot feature importance
    feature_names = tfidf_vectorizer.get_feature_names_out()
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.figure(figsize=(10, 8))
    plt.title("Feature importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return rf_classifier, tfidf_vectorizer
