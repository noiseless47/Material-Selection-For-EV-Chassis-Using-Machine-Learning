# Material Selection for EV Chassis - Web Application

## Overview
This web application provides a user-friendly interface for selecting materials for electric vehicle chassis using machine learning. The application uses Streamlit to create an interactive experience where users can input material properties and get instant predictions on suitability for EV chassis.

## Features
- **Interactive Material Prediction**: Input material properties and instantly see if they're suitable for EV chassis
- **Dataset Exploration**: View dataset statistics and sample materials
- **Model Performance Visualization**: See how different ML models perform in predicting material suitability
- **Material Property Analysis**: Visual comparison of material properties using radar charts

## Requirements
- Python 3.6+
- Required packages listed in requirements.txt

## Quick Start

### Option 1: Using the setup script (Windows)
1. Simply run the `setup.bat` file:
```
setup.bat
```
This will:
- Install all required packages
- Run the main script to generate models
- Start the Streamlit web application

### Option 2: Manual setup (All platforms)
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the main script to generate models:
```
python ev_chassis_material_selection.py
```

3. Start the Streamlit web application:
```
streamlit run app.py
```

## How to Use
1. **Material Prediction Tab**:
   - Enter material properties using the input fields
   - Click "Predict Suitability" to get results
   - View the radar chart showing property profile

2. **Dataset Info Tab**:
   - See statistics about the materials dataset
   - Browse sample materials

3. **Model Performance Tab**: 
   - View accuracy comparison of different models
   - See feature importance visualizations

## Application Architecture
The application consists of three main components:

1. **ev_chassis_ml.py**: Core ML module containing model training and prediction functions
2. **app.py**: Streamlit web interface
3. **ev_chassis_material_selection.py**: Original script for model training and evaluation

## Customization
- To add new material properties, update the PROPERTY_RANGES and COLUMN_INFO dictionaries in ev_chassis_ml.py
- To modify the UI, edit the app.py file
- To change default values, adjust the input fields in the material_prediction_ui function

## Troubleshooting
- If you see "Error: material.csv not found", ensure the data file is in the same directory
- If model prediction fails, try running the main script first: `python ev_chassis_material_selection.py`
- For visualization issues, ensure matplotlib is properly installed

## Credits
This application is based on research in material selection for EV chassis using machine learning techniques.

For more details on the research methodology and results, refer to:
https://iopscience.iop.org/article/10.1088/1742-6596/2601/1/012014 