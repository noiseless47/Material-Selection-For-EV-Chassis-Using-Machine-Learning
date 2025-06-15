#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material Selection for EV Chassis using Machine Learning - Core Module

This module contains the core ML functionality for EV chassis material selection
that can be imported and used by other applications.

Author: AI Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import os
import pickle
import joblib

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    from xgboost import XGBClassifier

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Define column names and their descriptions for the UI
COLUMN_INFO = {
    "Su": "Ultimate Tensile Strength (MPa)",
    "Sy": "Yield Strength (MPa)",
    "E": "Young's Modulus (MPa)",
    "G": "Shear Modulus (MPa)",
    "mu": "Poisson's Ratio",
    "Ro": "Density (kg/m³)",
    "Heat treatment": "Heat Treatment Process",
    "A5": "Elongation (%)",
    "Bhn": "Brinell Hardness",
    "HV": "Vickers Hardness"
}

# Define ranges for each property based on the dataset
PROPERTY_RANGES = {
    "Su": (100, 2300),  # MPa
    "Sy": (50, 2100),   # MPa
    "E": (70000, 210000), # MPa
    "G": (25000, 80000),  # MPa
    "mu": (0.2, 0.4),
    "Ro": (6000, 9000),  # kg/m³
}

# Define ideal property ranges for EV chassis
IDEAL_RANGES = {
    "Su": (350, 500),    # MPa
    "Sy": (250, 380),    # MPa
    "E": (190000, 210000), # MPa
    "G": (75000, 80000),   # MPa
    "mu": (0.28, 0.32),
    "Ro": (7000, 8000),    # kg/m³
}

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
    data = pd.read_csv(file_path)
    
    # Encode categorical variables
    le = preprocessing.LabelEncoder()
    columns = ["Material", "Use"]
    for col in columns:
        data[col] = le.fit_transform(data[col])
    
    # Split features and target
    y = data["Use"]
    X = data.drop("Use", axis=1)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    return X_train, X_test, y_train, y_test, data

def train_models(X_train, y_train):
    """
    Train all models on the provided data
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    y_train : Series
        Training target
        
    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    models = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=10000)
    lr_model.fit(X_train, y_train)
    models["Logistic Regression"] = lr_model
    
    # 2. Support Vector Machine
    svm_model = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    svm_model.fit(X_train, y_train)
    models["Support Vector Machine"] = svm_model
    
    # 3. K-Neighbors Classifier
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(X_train, y_train)
    models["K-Neighbors Classifier"] = knn_model
    
    # 4. Decision Tree
    dt_model = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)
    models["Decision Tree"] = dt_model
    
    # 5. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    models["Random Forest Classifier"] = rf_model
    
    # 6. XGBoost
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on test data
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : DataFrame
        Test features
    y_test : Series
        Test target
        
    Returns:
    --------
    dict
        Dictionary containing model accuracies
    """
    models_accuracy = {}
    
    for name, model in models.items():
        predictions = model.predict(X_test)
        models_accuracy[name] = accuracy_score(y_test, predictions)
    
    return models_accuracy

def get_best_model(models, models_accuracy):
    """
    Get the best performing model
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    models_accuracy : dict
        Dictionary of model accuracies
        
    Returns:
    --------
    tuple
        (best_model_name, best_model, accuracy)
    """
    best_model_name = max(models_accuracy, key=models_accuracy.get)
    return best_model_name, models[best_model_name], models_accuracy[best_model_name]

def get_feature_importance(model, feature_names):
    """
    Get feature importances from a model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
        
    Returns:
    --------
    list of tuples
        [(feature_name, importance), ...] sorted by importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = [(feature, importance) 
                             for feature, importance in zip(feature_names, importances)]
        return sorted(feature_importance, key=lambda x: x[1], reverse=True)
    else:
        return None

def save_models(models, directory='models'):
    """
    Save all trained models to disk
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    directory : str
        Directory to save models in
        
    Returns:
    --------
    None
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save each model
    for name, model in models.items():
        filename = os.path.join(directory, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, filename)
    
    print(f"All models saved to {directory}/ directory.")

def load_model(model_name, directory='models'):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_name : str
        Name of the model to load
    directory : str
        Directory where models are stored
        
    Returns:
    --------
    model
        Loaded model
    """
    filename = os.path.join(directory, f"{model_name.lower().replace(' ', '_')}.pkl")
    return joblib.load(filename)

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
    tuple
        (prediction, probability)
        prediction: 1 if suitable, 0 if not suitable
        probability: float between 0 and 1 indicating probability of suitability
    """
    # Create a feature vector with numerical values only
    X = []
    
    for feature in feature_names:
        # Skip the 'Material' feature for prediction as it needs encoding
        if feature == 'Material':
            # Just use 0 as a placeholder - this isn't ideal but works for demonstration
            X.append(0)
        elif feature in material_properties:
            # Make sure the value is a number
            try:
                value = float(material_properties[feature])
                X.append(value)
            except (ValueError, TypeError):
                X.append(0)
        else:
            # For missing features, use 0 as default
            X.append(0)
    
    X = np.array(X).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get prediction probability if the model supports it
    try:
        probability = model.predict_proba(X)[0][1]
    except:
        probability = None
    
    return prediction, probability

def create_model_comparison_chart(models_accuracy):
    """
    Create a plot comparing model accuracies
    
    Parameters:
    -----------
    models_accuracy : dict
        Dictionary of model accuracies
        
    Returns:
    --------
    matplotlib figure
        Figure showing model comparison
    """
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
    plt.title('Model Accuracy Comparison')
    plt.xlim(min(accuracy) - 0.05, 1.01)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracy):
        plt.text(acc + 0.005, bar.get_y() + bar.get_height()/2, 
                f"{acc:.4f}", va='center')
    
    plt.tight_layout()
    return plt.gcf()

def create_feature_importance_chart(model, feature_names, model_name):
    """
    Create a plot showing feature importances
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
        
    Returns:
    --------
    matplotlib figure or None
        Figure showing feature importances, or None if not available
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_feature_names)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    
    return plt.gcf()

def get_material_categories(data):
    """
    Extract material categories from the dataset
    
    Parameters:
    -----------
    data : DataFrame
        Dataset containing material information
        
    Returns:
    --------
    dict
        Dictionary with categories as keys and lists of materials as values
    """
    # Create category from first part of material name
    data['Category'] = data['Material'].apply(lambda x: x.split()[0] if isinstance(x, str) else "Other")
    
    # Create dictionary of categories and their materials
    categories = {}
    for category in sorted(data['Category'].unique()):
        materials = data[data['Category'] == category]['Material'].unique().tolist()
        categories[category] = materials
    
    return categories

def get_material_properties(data, material_name):
    """
    Get properties of a specific material
    
    Parameters:
    -----------
    data : DataFrame
        Dataset containing material information
    material_name : str
        Name of the material to get properties for
        
    Returns:
    --------
    dict
        Dictionary of material properties
    """
    # Get the material row
    material = data[data['Material'] == material_name]
    if len(material) == 0:
        return None
    
    # Convert to dictionary
    properties = material.iloc[0].to_dict()
    
    return properties

def analyze_material_database(data):
    """
    Analyze the material database to extract categories, types, and patterns
    
    Parameters:
    -----------
    data : DataFrame
        Dataset containing material information
        
    Returns:
    --------
    dict
        Dictionary with material analysis information
    """
    results = {}
    
    # Extract all unique materials
    materials = data['Material'].unique()
    results['total_materials'] = len(materials)
    
    # Extract material prefixes (first word)
    prefixes = []
    for material in materials:
        if isinstance(material, str):
            prefix = material.split()[0]
            prefixes.append(prefix)
    
    # Count occurrences of each prefix
    prefix_counts = {}
    for prefix in prefixes:
        if prefix in prefix_counts:
            prefix_counts[prefix] += 1
        else:
            prefix_counts[prefix] = 1
    
    results['material_prefixes'] = prefix_counts
    
    # Extract material types based on patterns
    material_types = {}
    for material in materials:
        if not isinstance(material, str):
            continue
            
        # Try to extract material type information
        material_parts = material.split()
        
        # Look for patterns like "Steel SAE 1020", "Copper Alloy C81300", etc.
        if "Steel" in material:
            # Extract steel grade if exists
            steel_type = next((part for part in material_parts if "SAE" in part), None)
            if steel_type:
                if steel_type not in material_types:
                    material_types[steel_type] = {"category": "Steel", "count": 0, "materials": []}
                material_types[steel_type]["count"] += 1
                material_types[steel_type]["materials"].append(material)
        elif "Grey cast iron" in material:
            iron_type = next((part for part in material_parts if "class" in part), None)
            if iron_type:
                if iron_type not in material_types:
                    material_types[iron_type] = {"category": "Cast Iron", "count": 0, "materials": []}
                material_types[iron_type]["count"] += 1
                material_types[iron_type]["materials"].append(material)
        elif "cast iron" in material.lower():
            if "cast iron" not in material_types:
                material_types["cast iron"] = {"category": "Cast Iron", "count": 0, "materials": []}
            material_types["cast iron"]["count"] += 1
            material_types["cast iron"]["materials"].append(material)
        elif "Brass" in material:
            brass_type = next((part for part in material_parts if part.startswith("C8")), None)
            if brass_type:
                if brass_type not in material_types:
                    material_types[brass_type] = {"category": "Brass", "count": 0, "materials": []}
                material_types[brass_type]["count"] += 1
                material_types[brass_type]["materials"].append(material)
        elif "Copper" in material:
            copper_type = next((part for part in material_parts if part.startswith("C8")), None)
            if copper_type:
                if copper_type not in material_types:
                    material_types[copper_type] = {"category": "Copper", "count": 0, "materials": []}
                material_types[copper_type]["count"] += 1
                material_types[copper_type]["materials"].append(material)
        elif "Bronze" in material:
            bronze_type = next((part for part in material_parts if part.startswith("C8")), None)
            if bronze_type:
                if bronze_type not in material_types:
                    material_types[bronze_type] = {"category": "Bronze", "count": 0, "materials": []}
                material_types[bronze_type]["count"] += 1
                material_types[bronze_type]["materials"].append(material)
        else:
            # Generic material type based on first word
            generic_type = material_parts[0]
            if generic_type not in material_types:
                material_types[generic_type] = {"category": "Other", "count": 0, "materials": []}
            material_types[generic_type]["count"] += 1
            material_types[generic_type]["materials"].append(material)
    
    results['material_types'] = material_types
    
    # Create material categories with descriptions
    categories = {}
    
    # Process steel types
    steel_types = {}
    for material in materials:
        if isinstance(material, str) and "Steel" in material and "SAE" in material:
            parts = material.split()
            sae_code = next((part for part in parts if "SAE" in part), None)
            if sae_code and len(sae_code) > 4:
                series = sae_code[4]  # Get the first digit after "SAE "
                if series not in steel_types:
                    steel_types[series] = []
                steel_types[series].append(material)
    
    # Create steel descriptions
    steel_descriptions = {
        "1": "Carbon Steel - Good general-purpose steel with moderate strength and machinability",
        "2": "Nickel Steel - Improved strength and toughness",
        "3": "Nickel-Chromium Steel - Good hardenability and strength",
        "4": "Molybdenum Steel - High hardenability and strength",
        "5": "Chromium Steel - Good corrosion resistance and wear resistance",
        "6": "Chromium-Vanadium Steel - High strength and toughness",
        "8": "Nickel-Chromium-Molybdenum Steel - High strength and toughness",
        "9": "Silicon-Manganese Steel - Good elasticity and fatigue resistance"
    }
    
    # Add steel type descriptions
    for series, materials_list in steel_types.items():
        description = steel_descriptions.get(series, "Alloy Steel")
        categories[f"SAE {series}xxx Series"] = {
            "materials": materials_list,
            "description": description,
            "count": len(materials_list)
        }
    
    # Add other material categories
    if any("Copper" in str(m) for m in materials):
        copper_materials = [m for m in materials if isinstance(m, str) and "Copper" in m]
        categories["Copper Alloys"] = {
            "materials": copper_materials,
            "description": "Copper alloys with good electrical and thermal conductivity",
            "count": len(copper_materials)
        }
    
    if any("Brass" in str(m) for m in materials):
        brass_materials = [m for m in materials if isinstance(m, str) and "Brass" in m]
        categories["Brass"] = {
            "materials": brass_materials,
            "description": "Copper-zinc alloys with good corrosion resistance and machinability",
            "count": len(brass_materials)
        }
    
    if any("Bronze" in str(m) for m in materials):
        bronze_materials = [m for m in materials if isinstance(m, str) and "Bronze" in m]
        categories["Bronze"] = {
            "materials": bronze_materials,
            "description": "Copper-tin alloys with good strength and wear resistance",
            "count": len(bronze_materials)
        }
    
    if any("cast iron" in str(m).lower() for m in materials):
        cast_iron_materials = [m for m in materials if isinstance(m, str) and "cast iron" in m.lower()]
        categories["Cast Iron"] = {
            "materials": cast_iron_materials,
            "description": "Iron-carbon alloys with good castability and vibration damping",
            "count": len(cast_iron_materials)
        }
    
    results['material_categories'] = categories
    
    return results


def get_material_info(material_name):
    """
    Get detailed information about a material based on its name
    
    Parameters:
    -----------
    material_name : str
        Name of the material
        
    Returns:
    --------
    dict
        Dictionary with material information
    """
    info = {"name": material_name}
    
    # Material category
    if not isinstance(material_name, str):
        return info
    
    # Extract description based on material name patterns
    if "Steel" in material_name:
        info["category"] = "Steel"
        
        # Extract steel type if possible
        parts = material_name.split()
        sae_code = next((part for part in parts if "SAE" in part), None)
        
        if sae_code and len(sae_code) > 4:
            series = sae_code[4]  # Get the first digit after "SAE "
            
            steel_descriptions = {
                "1": "Carbon Steel - Good general-purpose steel with moderate strength and machinability",
                "2": "Nickel Steel - Improved strength and toughness",
                "3": "Nickel-Chromium Steel - Good hardenability and strength",
                "4": "Molybdenum Steel - High hardenability and strength",
                "5": "Chromium Steel - Good corrosion resistance and wear resistance",
                "6": "Chromium-Vanadium Steel - High strength and toughness",
                "8": "Nickel-Chromium-Molybdenum Steel - High strength and toughness",
                "9": "Silicon-Manganese Steel - Good elasticity and fatigue resistance"
            }
            
            info["type"] = f"SAE {series}xxx Series"
            info["description"] = steel_descriptions.get(series, "Alloy Steel")
            
            # Add heat treatment info if in name
            if "normalized" in material_name.lower():
                info["heat_treatment"] = "Normalized - Heated and cooled in air for improved strength and grain structure"
            elif "annealed" in material_name.lower():
                info["heat_treatment"] = "Annealed - Heated and slowly cooled for improved ductility and machinability"
            elif "tempered" in material_name.lower():
                info["heat_treatment"] = "Tempered - Heat treated for balanced hardness, strength, and toughness"
                
    elif "Grey cast iron" in material_name:
        info["category"] = "Cast Iron"
        info["type"] = "Grey Cast Iron"
        info["description"] = "Contains graphite flakes, good vibration damping, excellent castability"
        
        # Extract class
        parts = material_name.split()
        iron_class = next((part for part in parts if "class" in part), None)
        if iron_class:
            info["subtype"] = iron_class
            
    elif "Malleable cast iron" in material_name:
        info["category"] = "Cast Iron"
        info["type"] = "Malleable Cast Iron"
        info["description"] = "Heat treated for improved ductility and machinability"
        
    elif "Nodular cast iron" in material_name:
        info["category"] = "Cast Iron"
        info["type"] = "Nodular Cast Iron"
        info["description"] = "Contains graphite as spheroids, higher strength and ductility than grey cast iron"
        
    elif "Copper" in material_name:
        info["category"] = "Copper Alloy"
        info["description"] = "Excellent electrical and thermal conductivity"
        
        # Extract copper alloy code
        parts = material_name.split()
        copper_code = next((part for part in parts if part.startswith("C")), None)
        if copper_code:
            info["type"] = copper_code
            
    elif "Brass" in material_name:
        info["category"] = "Brass"
        info["description"] = "Copper-zinc alloy, good corrosion resistance and machinability"
        
        # Extract brass type
        if "Red" in material_name:
            info["type"] = "Red Brass"
            info["description"] += ", higher copper content than yellow brass"
        elif "Yellow" in material_name:
            info["type"] = "Yellow Brass"
            info["description"] += ", standard brass with good formability"
        elif "Semi-red" in material_name:
            info["type"] = "Semi-red Brass"
            info["description"] += ", intermediate between red and yellow brass"
            
    elif "Bronze" in material_name:
        info["category"] = "Bronze"
        info["description"] = "Copper-tin alloy, good wear resistance"
        
        if "Manganese" in material_name:
            info["type"] = "Manganese Bronze"
            info["description"] = "High-strength bronze with additions of manganese and aluminum"
            
    else:
        # Generic material information
        info["category"] = material_name.split()[0]
        
    return info

if __name__ == "__main__":
    # This code runs when the script is executed directly
    print("This is a module for ML functionality. Import it in your application.")
    print("For the full script, run ev_chassis_material_selection.py") 