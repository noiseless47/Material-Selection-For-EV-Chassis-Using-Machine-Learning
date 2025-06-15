# Material Selection for EV Chassis using Machine Learning

## Overview
This Python script provides a complete machine learning solution for selecting optimal materials for electric vehicle chassis based on mechanical properties. It implements and compares multiple ML models to identify the most suitable materials.

## Features
- Data preprocessing and analysis
- Implementation of 6 machine learning models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - XGBoost
- Comprehensive model evaluation and comparison
- Feature importance analysis
- Visualization of results with charts and confusion matrices
- Material suitability prediction

## Requirements
- Python 3.6+
- Dependencies: numpy, pandas, matplotlib, scikit-learn, seaborn, xgboost, graphviz

## Installation
```bash
pip install numpy pandas matplotlib scikit-learn seaborn xgboost graphviz
```

## Usage
1. Ensure your data CSV file is in the same directory as the script
2. Run the script:
```bash
python ev_chassis_material_selection.py
```

The script will automatically:
- Load and preprocess the material data
- Train and evaluate all models
- Generate performance visualizations
- Analyze feature importance
- Show an example prediction

## Data Format
The expected CSV format should have the following columns:
- Material: Name of the material
- Su: Ultimate tensile strength (MPa)
- Sy: Yield strength (MPa)
- E: Young's modulus (MPa)
- G: Shear modulus (MPa)
- mu: Poisson's ratio
- Ro: Density (kg/m³)
- Use: Target variable (True/False) indicating if the material is suitable for EV chassis

## Output
The script produces:
- Detailed console output with model performance metrics
- Model comparison chart (model_comparison.png)
- Confusion matrices for top models
- Feature importance charts
- Decision tree visualization (decision_tree.png)

## Customization
You can modify the script to:
- Change the data source by updating the file path in the `load_and_preprocess_data()` function
- Adjust model parameters in the `main()` function
- Add new models or evaluation metrics
- Customize visualizations by modifying the plotting functions

## Example
For a new material with properties:
```python
sample_material = {
    "Su": 441,  # Ultimate Tensile Strength (MPa)
    "Sy": 346,  # Yield Strength (MPa)
    "E": 207000,  # Young's Modulus (MPa)
    "G": 79000,  # Shear Modulus (MPa)
    "mu": 0.3,  # Poisson's Ratio
    "Ro": 7860,  # Density (kg/m³)
}
```

The script will predict its suitability and display the confidence level for that prediction.

## Research Context
This implementation is based on research that achieved accuracy scores of:
- Linear Regression: 88.7%
- Support Vector Machine: 91.6%
- K-Neighbors Classifier: 98.7%
- Decision Tree: 99.7%
- Random Forest Classifier: 99.7%
- XGBoost: 99.4%

For more details, refer to the research paper: https://iopscience.iop.org/article/10.1088/1742-6596/2601/1/012014 