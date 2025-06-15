#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Material Selection for EV Chassis using Machine Learning - Web Interface

This script provides a Streamlit web interface for the EV chassis material selection system.

Author: AI Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import joblib
from PIL import Image
import io

# Import our ML module
import ev_chassis_ml as ml

# Set page configuration
st.set_page_config(
    page_title="EV Chassis Material Selection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .footer {
        font-size: 0.8rem;
        color: #666;
        text-align: center;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
        padding-top: 1rem;
    }
    .highlight {
        background-color: #f0f7fb;
        border-left: 5px solid #2196F3;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
    }
    .material-result {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: bold;
    }
    .material-suitable {
        background-color: #dff0d8;
        color: #3c763d;
        border: 1px solid #d6e9c6;
    }
    .material-unsuitable {
        background-color: #f2dede;
        color: #a94442;
        border: 1px solid #ebccd1;
    }
    .property-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODELS_DIR = 'models'
DATA_FILE = 'material.csv'

def load_data():
    """Load the dataset"""
    try:
        return pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Error: {DATA_FILE} not found. Please make sure it exists in the current directory.")
        return None

def ensure_models_exist():
    """Ensure model files exist, train them if not"""
    if not os.path.exists(MODELS_DIR):
        with st.spinner('Training models for the first time. This may take a moment...'):
            X_train, X_test, y_train, y_test, data = ml.load_and_preprocess_data(DATA_FILE)
            models = ml.train_models(X_train, y_train)
            ml.save_models(models, MODELS_DIR)
        st.success('Models trained and saved successfully!')
        return X_train.columns.tolist(), models
    
    # Get feature names from original data
    data = load_data()
    if data is None:
        return None, None
    
    feature_names = data.drop('Use', axis=1).columns.tolist()
    return feature_names, None

def display_header():
    """Display the app header"""
    st.markdown('<h1 class="main-header">🚗 Material Selection for EV Chassis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p class="highlight">
    This application helps select suitable materials for electric vehicle chassis 
    using machine learning algorithms trained on mechanical properties.
    </p>
    """, unsafe_allow_html=True)

def display_dataset_info():
    """Display information about the dataset"""
    st.markdown('<h2 class="sub-header">Dataset Information</h2>', unsafe_allow_html=True)
    
    data = load_data()
    if data is None:
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write(f"**Total materials:** {len(data)}")
        st.write(f"**Suitable materials:** {data['Use'].sum()}")
        st.write(f"**Unsuitable materials:** {len(data) - data['Use'].sum()}")
    
    with col2:
        # Calculate suitability percentage
        suitable_percentage = int((data['Use'].sum() / len(data)) * 100)
        unsuitable_percentage = 100 - suitable_percentage
        
        # Create a simple bar chart
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['Unsuitable', 'Suitable'], [unsuitable_percentage, suitable_percentage], color=['#FF6B6B', '#4CAF50'])
        ax.set_ylabel('Percentage')
        ax.set_title('Material Suitability Distribution')
        
        # Add values on bars
        for i, v in enumerate([unsuitable_percentage, suitable_percentage]):
            ax.text(i, v/2, f"{v}%", ha='center', fontweight='bold', color='white')
            
        st.pyplot(fig)

def display_model_stats():
    """Display model performance statistics"""
    st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
    
    # Check if model comparison chart exists
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png', caption='Model Accuracy Comparison', use_column_width=True)
    else:
        # Load data and train/evaluate models
        X_train, X_test, y_train, y_test, data = ml.load_and_preprocess_data(DATA_FILE)
        models = ml.train_models(X_train, y_train)
        models_accuracy = ml.evaluate_models(models, X_test, y_test)
        
        # Create and display model comparison chart
        fig = ml.create_model_comparison_chart(models_accuracy)
        st.pyplot(fig)
        
        # Create decision tree visualization if not exists
        if 'Decision Tree' in models and not os.path.exists('feature_importance_decision_tree.png'):
            fig = ml.create_feature_importance_chart(models['Decision Tree'], X_train.columns, 'Decision Tree')
            if fig:
                st.pyplot(fig)

def material_prediction_ui(feature_names):
    """UI for material property input and prediction"""
    st.markdown('<h2 class="sub-header">Predict Material Suitability</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    Enter the mechanical properties of your material to predict if it's suitable for EV chassis.
    Use the sliders to adjust values within typical ranges, or enter precise values directly.
    """)
    
    # Create a form for material properties
    with st.form(key='material_form'):
        material_props = {}
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strength Properties")
            material_props['Su'] = st.number_input(
                f"Ultimate Tensile Strength (MPa)", 
                min_value=ml.PROPERTY_RANGES['Su'][0],
                max_value=ml.PROPERTY_RANGES['Su'][1],
                value=500)
                
            material_props['Sy'] = st.number_input(
                f"Yield Strength (MPa)", 
                min_value=ml.PROPERTY_RANGES['Sy'][0],
                max_value=ml.PROPERTY_RANGES['Sy'][1],
                value=350)
            
            material_props['Ro'] = st.number_input(
                f"Density (kg/m³)", 
                min_value=ml.PROPERTY_RANGES['Ro'][0],
                max_value=ml.PROPERTY_RANGES['Ro'][1],
                value=7800)
            
        with col2:
            st.subheader("Elastic Properties")
            material_props['E'] = st.number_input(
                f"Young's Modulus (MPa)", 
                min_value=ml.PROPERTY_RANGES['E'][0],
                max_value=ml.PROPERTY_RANGES['E'][1],
                value=200000)
                
            material_props['G'] = st.number_input(
                f"Shear Modulus (MPa)", 
                min_value=ml.PROPERTY_RANGES['G'][0],
                max_value=ml.PROPERTY_RANGES['G'][1],
                value=75000)
                
            material_props['mu'] = st.slider(
                f"Poisson's Ratio", 
                min_value=ml.PROPERTY_RANGES['mu'][0],
                max_value=ml.PROPERTY_RANGES['mu'][1],
                value=0.3,
                step=0.01)
        
        submit_button = st.form_submit_button(label='Predict Suitability')
    
    # Handle prediction
    if submit_button:
        # Get the best model
        try:
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            # Load the Decision Tree model (best performer)
            best_model_path = os.path.join(MODELS_DIR, 'decision_tree.pkl')
            if not os.path.exists(best_model_path):
                best_model_path = os.path.join(MODELS_DIR, model_files[0])
            
            model = joblib.load(best_model_path)
            
            # Make prediction
            with st.spinner('Analyzing material properties...'):
                time.sleep(1)  # Slight delay for better UX
                prediction, probability = ml.predict_material_suitability(model, material_props, feature_names)
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="material-result material-suitable">
                    ✅ This material is SUITABLE for EV chassis!
                    {f"<br>Confidence: {probability:.2%}" if probability is not None else ""}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="material-result material-unsuitable">
                    ❌ This material is NOT SUITABLE for EV chassis.
                    {f"<br>Confidence: {1-probability:.2%}" if probability is not None else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Show material property radar chart
            display_material_radar_chart(material_props)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Try running the main script first: `python ev_chassis_material_selection.py`")

def display_material_radar_chart(material_props):
    """Display radar chart of material properties compared to typical ranges"""
    st.subheader("Material Property Analysis")
    
    # Normalize values between 0 and 1 for radar chart
    normalized_props = {}
    for prop, value in material_props.items():
        if prop in ml.PROPERTY_RANGES:
            min_val, max_val = ml.PROPERTY_RANGES[prop]
            normalized_props[prop] = (value - min_val) / (max_val - min_val)
    
    # Create radar chart
    categories = list(normalized_props.keys())
    values = list(normalized_props.values())
    
    # Complete the loop
    values += values[:1]
    categories += categories[:1]
    
    # Create the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot the values
    ax.plot(np.linspace(0, 2*np.pi, len(values)), values, 'o-', linewidth=2, label='Your Material')
    ax.fill(np.linspace(0, 2*np.pi, len(values)), values, alpha=0.25)
    
    # Set the angles for each category
    ax.set_xticks(np.linspace(0, 2*np.pi, len(categories)-1, endpoint=False))
    ax.set_xticklabels([f"{cat}\n({ml.COLUMN_INFO[cat]})" for cat in categories[:-1]])
    
    # Set the radial limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'])
    
    plt.title('Material Property Profile\n(relative to typical ranges)', pad=20)
    st.pyplot(fig)

def predefined_materials_ui(feature_names):
    """UI for selecting predefined materials and displaying their suitability"""
    st.markdown('<h2 class="sub-header">Predefined Materials Selection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="highlight">
    Select from standardized materials to see their properties and suitability for EV chassis.
    </p>
    """, unsafe_allow_html=True)
    
    # Load the data
    data = load_data()
    if data is None:
        return
    
    # Process data for display
    # Convert boolean/integer 'Use' to string for better display
    if 'Use' in data.columns:
        if data['Use'].dtype == bool or data['Use'].dtype == int:
            data['Suitable'] = data['Use'].apply(lambda x: "✅ YES" if x else "❌ NO")
    
    # Analyze the database to get material categories
    try:
        # Analyze the database to extract material information
        material_analysis = ml.analyze_material_database(data)
        material_categories = material_analysis['material_categories']
        material_prefixes = material_analysis['material_prefixes']
        
        # Sort prefixes by count for organized selection
        sorted_prefixes = sorted(material_prefixes.items(), key=lambda x: x[1], reverse=True)
        prefix_categories = [prefix for prefix, count in sorted_prefixes]
    except Exception as e:
        st.error(f"Error analyzing material database: {e}")
        # Fallback to simple categorization
        data['Category'] = data['Material'].apply(lambda x: x.split()[0] if isinstance(x, str) else "Other")
        prefix_categories = sorted(data['Category'].unique())
        material_categories = {}
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Material")
        
        # Add filter options
        filter_option = st.radio(
            "Filter materials by suitability:",
            ["All Materials", "Only Suitable", "Only Unsuitable"],
            horizontal=True
        )
        
        # Filter data based on selection
        if filter_option == "Only Suitable":
            data = data[data['Use'] == True]
        elif filter_option == "Only Unsuitable":
            data = data[data['Use'] == False]
        
        # Display material categories with counts
        st.subheader("Material Categories")
        
        try:
            expander_states = {}
            
            # Display SAE steel series categories in expanders
            for category_name in [cat for cat in material_categories if "SAE" in cat]:
                try:
                    category_info = material_categories[category_name]
                    # Filter by suitability if needed
                    if filter_option == "Only Suitable":
                        materials_in_category = [m for m in category_info["materials"] 
                                                if m in data['Material'].values]
                    elif filter_option == "Only Unsuitable":
                        materials_in_category = [m for m in category_info["materials"] 
                                                if m in data['Material'].values]
                    else:
                        materials_in_category = category_info["materials"]
                        
                    if materials_in_category:
                        expander = st.expander(f"{category_name} ({len(materials_in_category)} materials)")
                        with expander:
                            st.write(f"**{category_info['description']}**")
                            
                            if "heat_treatments" in category_info:
                                st.write("Common heat treatments:")
                                for treatment, desc in category_info["heat_treatments"].items():
                                    st.write(f"- **{treatment}**: {desc}")
                except Exception as e:
                    st.error(f"Error displaying category {category_name}: {e}")
            
            # Special category for cast iron
            if "Cast Iron" in material_categories:
                try:
                    category_info = material_categories["Cast Iron"]
                    # Filter by suitability if needed
                    if filter_option == "Only Suitable":
                        materials_in_category = [m for m in category_info["materials"] 
                                                if m in data['Material'].values]
                    elif filter_option == "Only Unsuitable":
                        materials_in_category = [m for m in category_info["materials"] 
                                                if m in data['Material'].values]
                    else:
                        materials_in_category = category_info["materials"]
                        
                    if materials_in_category:
                        expander = st.expander(f"Cast Iron ({len(materials_in_category)} materials)")
                        with expander:
                            st.write(f"**{category_info['description']}**")
                except Exception as e:
                    st.error(f"Error displaying Cast Iron category: {e}")
            
            # Non-ferrous metals
            for category_name in ["Copper Alloys", "Brass", "Bronze"]:
                if category_name in material_categories:
                    try:
                        category_info = material_categories[category_name]
                        # Filter by suitability if needed
                        if filter_option == "Only Suitable":
                            materials_in_category = [m for m in category_info["materials"] 
                                                    if m in data['Material'].values]
                        elif filter_option == "Only Unsuitable":
                            materials_in_category = [m for m in category_info["materials"] 
                                                    if m in data['Material'].values]
                        else:
                            materials_in_category = category_info["materials"]
                            
                        if materials_in_category:
                            expander = st.expander(f"{category_name} ({len(materials_in_category)} materials)")
                            with expander:
                                st.write(f"**{category_info['description']}**")
                    except Exception as e:
                        st.error(f"Error displaying {category_name} category: {e}")
        except Exception as e:
            st.warning(f"Error displaying material categories: {e}")
        
        # Select material prefix/category
        st.subheader("Select Material")
        
        # Calculate categories with their counts in the filtered data
        data['Category'] = data['Material'].apply(lambda x: x.split()[0] if isinstance(x, str) else "Other")
        category_counts = data['Category'].value_counts().to_dict()
        category_options = [f"{cat} ({category_counts.get(cat, 0)})" for cat in sorted(category_counts.keys())]
        
        selected_category_with_count = st.selectbox(
            "Material Category",
            options=category_options,
            index=0 if category_options and "ANSI" in category_options[0] else 0
        )
        
        # Extract the actual category name without count
        selected_category = selected_category_with_count.split(" (")[0]
        
        # Then filter materials by category
        filtered_materials = data[data['Category'] == selected_category]
        material_names = filtered_materials['Material'].unique()
        
        selected_material_name = st.selectbox(
            "Material",
            options=material_names
        )
        
        # Get the selected material details
        selected_material = data[data['Material'] == selected_material_name].iloc[0]
        
        # Get detailed material info
        try:
            material_info = ml.get_material_info(selected_material_name)
            
            # Display material type information if available
            if "type" in material_info:
                st.info(f"**Material Type**: {material_info['type']}")
                
            # Display heat treatment if available
            if "heat_treatment" in material_info:
                st.info(f"**Heat Treatment**: {material_info['heat_treatment']}")
            elif 'Heat treatment' in data.columns and pd.notna(selected_material.get('Heat treatment')):
                st.info(f"**Heat Treatment**: {selected_material['Heat treatment']}")
                
            # Display description if available
            if "description" in material_info:
                st.success(f"**Description**: {material_info['description']}")
        except Exception as e:
            st.warning(f"Could not retrieve detailed material info: {e}")
            # If we can't get material info, at least display heat treatment if available
            if 'Heat treatment' in data.columns and pd.notna(selected_material.get('Heat treatment')):
                st.info(f"**Heat Treatment**: {selected_material['Heat treatment']}")
    
    with col2:
        st.subheader("Material Properties")
        
        # Create a styled container for properties
        with st.container():
            # Display each property in a clean format with proper styling
            for prop in ml.COLUMN_INFO.keys():
                if prop in selected_material and pd.notna(selected_material[prop]):
                    st.write(f"**{ml.COLUMN_INFO[prop]}:** {selected_material[prop]}")
        
        # Display material property metrics in a more visual way
        key_properties = ['Ultimate tensile strength', 'Yield Strength', "Young's modulus", 'Density']
        valid_props = [p for p in key_properties if p in selected_material and pd.notna(selected_material[p])]
        
        # Only create columns if we have valid properties
        if valid_props:
            cols = st.columns(len(valid_props))
            
            for i, prop in enumerate(valid_props):
                with cols[i]:
                    # Add units based on property
                    unit = "(MPa)" if "strength" in prop.lower() or "modulus" in prop.lower() else ""
                    unit = "(kg/m³)" if "density" in prop.lower() else unit
                    st.metric(f"{prop} {unit}", f"{selected_material[prop]}")
    
    # Display suitability information prominently
    is_suitable = bool(selected_material.get('Use', False))
    
    if is_suitable:
        st.markdown("""
        <div style="padding: 15px; background-color: #dff0d8; border: 1px solid #d6e9c6; 
                   border-radius: 5px; color: #3c763d; font-weight: bold; text-align: center; margin: 15px 0;">
            ✅ This material is SUITABLE for EV chassis!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 15px; background-color: #f2dede; border: 1px solid #ebccd1; 
                   border-radius: 5px; color: #a94442; font-weight: bold; text-align: center; margin: 15px 0;">
            ❌ This material is NOT SUITABLE for EV chassis.
        </div>
        """, unsafe_allow_html=True)
    
    # Get similar materials section
    st.subheader("Similar Materials")
    
    # Extract numerical features for similarity comparison
    numerical_cols = [col for col in ml.COLUMN_INFO.keys() if col in data.columns]
    
    if len(numerical_cols) > 0:
        # Get values of selected material
        selected_values = selected_material[numerical_cols].values
        
        # Function to calculate similarity (Euclidean distance)
        def calculate_similarity(row):
            return np.sqrt(sum((row[numerical_cols].values - selected_values)**2))
        
        # Calculate similarity for all materials
        data_with_similarity = data.copy()
        data_with_similarity['similarity'] = data.apply(calculate_similarity, axis=1)
        
        # Get similar materials (exclude the selected one)
        similar_materials = data_with_similarity[data_with_similarity['Material'] != selected_material_name]
        similar_materials = similar_materials.sort_values('similarity').head(5)
        
        # Display similar materials with their properties
        for i, material in similar_materials.iterrows():
            with st.expander(f"{material['Material']} - {'✅ Suitable' if bool(material.get('Use', False)) else '❌ Not Suitable'}"):
                props_text = ""
                for prop in numerical_cols:
                    if pd.notna(material[prop]):
                        props_text += f"**{ml.COLUMN_INFO[prop]}:** {material[prop]}  \n"
                st.markdown(props_text)
    else:
        st.info("Not enough numerical properties to find similar materials.")
    
    # Analysis and comparison with radar chart
    st.subheader("Material Property Analysis")
    
    try:
        # Create material properties dictionary for radar chart
        material_props = {}
        for prop in ml.PROPERTY_RANGES.keys():
            if prop in selected_material and pd.notna(selected_material[prop]):
                material_props[prop] = float(selected_material[prop])
        
        # Display radar chart
        if len(material_props) >= 3:  # Need at least 3 properties for a meaningful chart
            display_material_radar_chart(material_props)
        else:
            st.info("Not enough properties to create a radar chart.")
    except Exception as e:
        st.error(f"Could not create radar chart: {e}")
        
    # Compare with ideal properties
    st.subheader("Comparison with Ideal Properties")
    
    # Use ideal ranges from the ML module
    ideal_ranges = ml.IDEAL_RANGES
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    properties = []
    values = []
    colors = []
    
    for prop in ideal_ranges.keys():
        if prop in selected_material and pd.notna(selected_material[prop]):
            prop_val = float(selected_material[prop])
            min_val, max_val = ideal_ranges[prop]
            properties.append(prop)
            values.append(prop_val)
            
            # Determine color based on whether property is in ideal range
            if min_val <= prop_val <= max_val:
                colors.append('#4CAF50')  # Green for good
            elif prop_val < min_val * 0.8 or prop_val > max_val * 1.2:
                colors.append('#F44336')  # Red for bad
            else:
                colors.append('#FFC107')  # Yellow for borderline
    
    # Create bar chart
    if properties:
        y_pos = np.arange(len(properties))
        ax.barh(y_pos, values, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{p} ({ml.COLUMN_INFO[p]})" for p in properties])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title('Material Properties vs. Ideal Range')
        
        # Add ideal range bands
        for i, prop in enumerate(properties):
            min_val, max_val = ideal_ranges[prop]
            ax.axvspan(min_val, max_val, alpha=0.2, color='green', ymin=(len(properties)-1-i)/len(properties), ymax=(len(properties)-i)/len(properties))
            
        st.pyplot(fig)
    else:
        st.info("Not enough data to create comparison chart.")

def main():
    """Main function to run the app"""
    display_header()
    
    # Ensure models exist
    feature_names, _ = ensure_models_exist()
    if feature_names is None:
        st.error("Cannot continue without proper data. Please check the data file.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Material Prediction", "Predefined Materials", "Dataset Info", "Model Performance"])
    
    with tab1:
        material_prediction_ui(feature_names)
    
    with tab2:
        predefined_materials_ui(feature_names)
    
    with tab3:
        display_dataset_info()
        
        # Display a sample of the dataset
        st.subheader("Sample Materials")
        data = load_data()
        if data is not None:
            st.dataframe(data.head(10))
    
    with tab4:
        display_model_stats()
        
        # Display decision tree visualization if it exists
        if os.path.exists('decision_tree.png'):
            st.image('decision_tree.png', caption='Decision Tree Visualization', use_column_width=True)
        elif os.path.exists('decision_tree.dot'):
            st.info("Decision tree dot file exists but couldn't be rendered. Install Graphviz to view the visualization.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Material Selection for EV Chassis using Machine Learning<br>
        Developed for electric vehicle design and material selection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 