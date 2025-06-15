@echo off
echo ===================================================
echo Setting up environment for EV Chassis Material Selection
echo ===================================================
echo.

echo Installing required Python packages...
pip install numpy pandas matplotlib scikit-learn seaborn xgboost graphviz

echo.
echo ===================================================
echo To visualize decision trees, you need to install Graphviz:
echo 1. Download Graphviz from https://graphviz.org/download/
echo 2. During installation, check "Add Graphviz to the system PATH"
echo.
echo After installing Graphviz, you may need to restart your command prompt.
echo ===================================================
echo.

echo Setup complete!
echo Run the script with: python ev_chassis_material_selection.py
echo.
pause 