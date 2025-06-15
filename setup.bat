@echo off
echo ===================================================
echo Setting up EV Chassis Material Selection Web Application
echo ===================================================
echo.

echo Step 1: Installing required Python packages...
pip install -r requirements.txt

echo.
echo Step 2: Running the main script to generate models...
python ev_chassis_material_selection.py

echo.
echo Step 3: Starting the Streamlit web application...
echo.
echo IMPORTANT: The web interface will open in your default browser.
echo When you want to close the application, press Ctrl+C in this window.
echo.
streamlit run app.py

pause 