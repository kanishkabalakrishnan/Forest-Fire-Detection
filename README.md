                                                                          Forest Fire Detection Project
Overview
The Forest Fire Detection project aims to leverage Convolutional Neural Networks (CNNs) for real-time detection and intensity assessment of forest fires using live video feeds. The application also provides weather information based on the user's location, helping in the comprehensive assessment of fire risk and management.

Features
Fire Detection: The model classifies frames as either "Fire" or "No Fire" using a CNN-based image classification model.
Fire Intensity Assessment: Depending on the detected fire, the system categorizes the fire intensity into "Low", "Medium", or "High" based on pixel count and time of day.
Weather Information: Utilizes OpenWeatherAPI to fetch and display current weather conditions such as temperature, humidity, wind direction, and wind speed.
Model Details
The project uses a pre-trained baseline model for fire detection. The model was built using TensorFlow and trained on a dataset of fire and non-fire images to accurately classify the presence of fire.

Dependencies
OpenCV
TensorFlow
NumPy
Streamlit
OpenWeatherAPI
Requests
Setup Instructions
Install Dependencies:
          pip install opencv-python tensorflow numpy streamlit requests
Download Pre-trained Model: Ensure the pre-trained model baseline_model.hdf5 is placed in the ./models/ directory.

Run Streamlit Application:
   streamlit run app.py
   The Streamlit application captures live video, processes each frame, and displays the prediction and weather data. If fire is detected, it also provides an intensity assessment.
Conclusion
This project showcases the integration of machine learning and real-time data processing to address a critical issue of forest fire detection. It demonstrates the practical application of CNNs in image classification and the utility of weather data in enhancing model predictions.

The project's success lies in its ability to process live video feeds, accurately detect fire, and provide crucial contextual information like fire intensity and weather conditions, aiding in timely interventions and potentially saving lives and property.
