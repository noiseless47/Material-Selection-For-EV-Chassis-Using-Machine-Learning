#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material Selection for EV Chassis using Machine Learning

This script implements multiple machine learning models to select optimal materials
for electric vehicle chassis based on their mechanical properties.

Models implemented:
- Logistic Regression
- Support Vector Machine
- K-Neighbors Classifier
- Decision Tree
- Random Forest Classifier
- XGBoost

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    from xgboost import XGBClassifier

try:
    import graphviz
except ImportError:
    print("Graphviz not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "graphviz"])
    import graphviz

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_and_preprocess_data(file_path='material.csv'):
    """
    Load and preprocess the material data
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing material data
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and testing data splits
    """
    print("Loading data from:", file_path)
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(f"Total number of samples: {len(data)}")
    print(f"Number of features: {len(data.columns) - 2}")  # Excluding Material and Use
    
    # Check for missing values
    missing_values = data.isna().sum()
    if missing_values.sum() > 0:
        print("\nMissing values detected:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values detected in the dataset")
    
    # Encode categorical variables
    le = preprocessing.LabelEncoder()
    columns = ["Material", "Use"]
    for col in columns:
        data[col] = le.fit_transform(data[col])
    
    # Split features and target
    y = data["Use"]
    X = data.drop("Use", axis=1)
    
    # Display class distribution
    class_distribution = y.value_counts()
    print("\nClass Distribution:")
    print(class_distribution)
    print(f"Positive class ratio: {class_distribution[1] / len(y):.2%}")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f'\nTraining data has {len(X_train)} rows')
    print(f'Test data has {len(X_test)} rows')
    
    return X_train, X_test, y_train, y_test, data


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model performance
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model to evaluate
    X_test : DataFrame
        Test features
    y_test : Series
        Test target
    model_name : str
        Name of the model for display purposes
        
    Returns:
    --------
    float
        Accuracy score
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy of {model_name} is: {accuracy:.4f}')
    
    # Generate detailed classification report
    report = classification_report(y_test, predictions)
    print(f"\n{model_name} Classification Report:\n{report}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    return accuracy, cm, predictions


def visualize_model_comparison(models_accuracy, title="Model Comparison"):
    """
    Visualize the comparison of different models' performance
    
    Parameters:
    -----------
    models_accuracy : dict
        Dictionary containing model names and their accuracy scores
    title : str
        Title for the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        models = list(models_accuracy.keys())
        accuracy = list(models_accuracy.values())
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracy)
        models = [models[i] for i in sorted_indices]
        accuracy = [accuracy[i] for i in sorted_indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        
        bars = plt.barh(models, accuracy, color=colors)
        plt.xlabel('Accuracy')
        plt.title(title)
        plt.xlim(min(accuracy) - 0.05, 1.01)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracy):
            plt.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                    f"{acc:.4f}", va='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Model comparison chart saved as 'model_comparison.png'")
    except Exception as e:
        print(f"Error creating model comparison visualization: {e}")
        print("Continuing with program execution...")


def visualize_confusion_matrix(cm, model_name, labels=["Not Suitable", "Suitable"]):
    """
    Visualize the confusion matrix
    
    Parameters:
    -----------
    cm : array
        Confusion matrix
    model_name : str
        Name of the model
    labels : list
        Class labels
    """
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved as 'confusion_matrix_{model_name.replace(' ', '_').lower()}.png'")
    except Exception as e:
        print(f"Error creating confusion matrix visualization: {e}")
        print("Continuing with program execution...")


def visualize_decision_tree(model, feature_names, class_names=["Not Suitable", "Suitable"]):
    """
    Visualize the decision tree model
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained Decision Tree model
    feature_names : list
        List of feature names
    class_names : list
        List of class names
    """
    try:
        # Generate the graphviz representation of the decision tree
        dot_data = export_graphviz(
            model, 
            feature_names=feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True, 
            out_file=None
        )
        
        # Save the dot data to a file directly
        with open('decision_tree.dot', 'w') as f:
            f.write(dot_data)
        print("Decision tree dot file saved as 'decision_tree.dot'")
        
        try:
            # Attempt to render the graph if Graphviz is installed
            graph = graphviz.Source(dot_data)
            graph.render('decision_tree', format='png', cleanup=True)
            print("Decision tree visualization saved as 'decision_tree.png'")
        except Exception as e:
            print(f"Could not render decision tree visualization: {e}")
            print("To visualize the decision tree, please install Graphviz software from: https://graphviz.org/download/")
            print("After installing, make sure to add the Graphviz bin directory to your PATH environment variable.")
            print("You can still use the saved 'decision_tree.dot' file with an online Graphviz viewer.")
    except ValueError as e:
        print(f"Error in visualizing decision tree: {e}")
        print(f"Number of features in model: {model.n_features_in_}")
        print(f"Feature names provided: {feature_names}")
        print("Continuing with program execution...")


def feature_importance_analysis(model, feature_names, model_name):
    """
    Analyze and visualize feature importance
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model for display purposes
    """
    try:
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Print feature ranking
        print(f"\n{model_name} Feature Ranking:")
        for i, feature in enumerate(sorted_feature_names):
            print(f"{i+1}. {feature}: {sorted_importances[i]:.4f}")
        
        try:
            # Visualize feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_feature_names)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Feature importance chart saved as 'feature_importance_{model_name.replace(' ', '_').lower()}.png'")
        except Exception as e:
            print(f"Error saving feature importance visualization: {e}")
            print("Continuing with program execution...")
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        print("Continuing with program execution...")


def predict_material_suitability(model, material_properties, feature_names):
    """
    Predicts if a material is suitable for EV chassis based on its properties
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model to use for prediction
    material_properties : dict
        Dictionary containing material properties
    feature_names : list
        List of feature names in the correct order for the model
        
    Returns:
    --------
    int
        1 if suitable, 0 if not suitable
    float
        Probability of being suitable
    """
    # Create a feature vector with numerical values only
    X = []
    
    for feature in feature_names:
        # Skip the 'Material' feature for prediction as it needs encoding
        if feature == 'Material':
            # Just use 0 as a placeholder - this isn't ideal but will work for demonstration
            X.append(0)
            print(f"Note: Feature 'Material' is encoded as 0 for prediction")
        elif feature in material_properties:
            # Make sure the value is a number
            try:
                value = float(material_properties[feature])
                X.append(value)
            except (ValueError, TypeError):
                print(f"Warning: Feature '{feature}' value '{material_properties[feature]}' is not numeric, using 0")
                X.append(0)
        else:
            # For missing features, use 0 as default
            X.append(0)
            print(f"Warning: Feature '{feature}' not provided, using default value of 0")
    
    X = np.array(X).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get prediction probability if the model supports it
    try:
        probability = model.predict_proba(X)[0][1]
    except:
        probability = None
    
    return prediction, probability


def main():
    """Main function to run the material selection analysis"""
    print("=" * 80)
    print("Material Selection for EV Chassis using Machine Learning")
    print("=" * 80)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, data = load_and_preprocess_data()
    
    # List numerical features (excluding 'Material')
    numerical_features = X_train.columns.tolist()
    
    # Dictionary to store model accuracies
    models_accuracy = {}
    confusion_matrices = {}
    
    # Train and evaluate models
    print("\n" + "=" * 80)
    print("Training and Evaluating Machine Learning Models")
    print("=" * 80)
    
    # 1. Logistic Regression
    print("\n1. Logistic Regression")
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)
    lr_model.fit(X_train, y_train)
    lr_acc, lr_cm, _ = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    models_accuracy["Logistic Regression"] = lr_acc
    confusion_matrices["Logistic Regression"] = lr_cm
    
    # 2. Support Vector Machine
    print("\n2. Support Vector Machine")
    svm_model = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    svm_model.fit(X_train, y_train)
    svm_acc, svm_cm, _ = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")
    models_accuracy["Support Vector Machine"] = svm_acc
    confusion_matrices["Support Vector Machine"] = svm_cm
    
    # 3. K-Neighbors Classifier
    print("\n3. K-Neighbors Classifier")
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, y_train)
    knn_acc, knn_cm, _ = evaluate_model(knn_model, X_test, y_test, "K-Neighbors Classifier")
    models_accuracy["K-Neighbors Classifier"] = knn_acc
    confusion_matrices["K-Neighbors Classifier"] = knn_cm
    
    # 4. Decision Tree
    print("\n4. Decision Tree")
    dt_model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)
    dt_acc, dt_cm, _ = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    models_accuracy["Decision Tree"] = dt_acc
    confusion_matrices["Decision Tree"] = dt_cm
    
    # Visualize decision tree
    visualize_decision_tree(dt_model, numerical_features)
    
    # Analyze feature importance for Decision Tree
    feature_importance_analysis(dt_model, numerical_features, "Decision Tree")
    
    # 5. Random Forest Classifier
    print("\n5. Random Forest Classifier")
    rf_model = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    rf_acc, rf_cm, _ = evaluate_model(rf_model, X_test, y_test, "Random Forest Classifier")
    models_accuracy["Random Forest Classifier"] = rf_acc
    confusion_matrices["Random Forest Classifier"] = rf_cm
    
    # Analyze feature importance for Random Forest
    feature_importance_analysis(rf_model, numerical_features, "Random Forest")
    
    # 6. XG Boost
    print("\n6. XGBoost")
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    xgb_acc, xgb_cm, _ = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    models_accuracy["XGBoost"] = xgb_acc
    confusion_matrices["XGBoost"] = xgb_cm
    
    # Analyze feature importance for XGBoost
    feature_importance_analysis(xgb_model, numerical_features, "XGBoost")
    
    # Visualize model comparison
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    visualize_model_comparison(models_accuracy, "Model Accuracy Comparison")
    
    # Visualize confusion matrices for the best models
    for model_name, cm in confusion_matrices.items():
        if models_accuracy[model_name] > 0.95:  # Only visualize the best models
            visualize_confusion_matrix(cm, model_name)
    
    # Select the best model for prediction
    best_model_name = max(models_accuracy, key=models_accuracy.get)
    best_model_acc = models_accuracy[best_model_name]
    print(f"\nBest Model: {best_model_name} with accuracy: {best_model_acc:.4f}")
    
    # Find the actual best model object
    if best_model_name == "Logistic Regression":
        best_model = lr_model
    elif best_model_name == "Support Vector Machine":
        best_model = svm_model
    elif best_model_name == "K-Neighbors Classifier":
        best_model = knn_model
    elif best_model_name == "Decision Tree":
        best_model = dt_model
    elif best_model_name == "Random Forest Classifier":
        best_model = rf_model
    elif best_model_name == "XGBoost":
        best_model = xgb_model
    
    # Example usage of material prediction
    print("\n" + "=" * 80)
    print("Example Material Prediction")
    print("=" * 80)
    
    # Sample material properties (ANSI Steel SAE 1020 normalized)
    sample_material = {
        # Material name is included for reference but will be handled specially in prediction
        "Material": "ANSI Steel SAE 1020 normalized",
        "Su": 441,  # Ultimate Tensile Strength (MPa)
        "Sy": 346,  # Yield Strength (MPa)
        "E": 207000,  # Young's Modulus (MPa)
        "G": 79000,  # Shear Modulus (MPa)
        "mu": 0.3,  # Poisson's Ratio
        "Ro": 7860,  # Density (kg/m³)
    }
    
    # Print sample material properties
    print("\nSample Material Properties:")
    for prop, value in sample_material.items():
        print(f"{prop}: {value}")
    
    try:
        # Predict suitability
        prediction, probability = predict_material_suitability(best_model, sample_material, numerical_features)
        
        print(f"\nPrediction: {'Suitable' if prediction == 1 else 'Not Suitable'} for EV chassis")
        if probability is not None:
            print(f"Confidence: {probability:.2%}")
    except Exception as e:
        print(f"\nError in prediction: {e}")
        print("Please check the material properties and try again.")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_model_acc:.4f}")
    print("\nThis project analyzed various materials for EV chassis selection using multiple ML models.")
    print("The models were trained on material mechanical properties and evaluated for accuracy.")
    print(f"The best performing model was {best_model_name} with {best_model_acc:.2%} accuracy.")
    print("\nThe most important features for material selection were:")
    
    # Display top features based on best model
    try:
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:3]  # Get top 3 features
            sorted_features = [numerical_features[i] for i in indices]
            sorted_importances = [importances[i] for i in indices]
            
            for i, (feature, importance) in enumerate(zip(sorted_features, sorted_importances)):
                print(f"{i+1}. {feature}: {importance:.2%}")
        else:
            print("Feature importance analysis not available for this model type.")
    except Exception as e:
        print(f"Could not display feature importances: {e}")
        
    print("\nCheck the generated visualization files for more details on model performance.")


if __name__ == "__main__":
    main() 