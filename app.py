import streamlit as st
import cv2
import numpy as np
import tempfile
import math
import tensorflow as tf

from weather import get_weather, get_time
from intensity import get_pixel_count, find_intensity

def predicting(image, model, class_names=['Fire', 'No Fire']):
    image = load_and_prep(image)
    image = tf.expand_dims(image, axis=0)  # Keep as float, no need to cast to tf.int16
    preds = model.predict(image)
    print(preds)
    if(math.isnan(preds[0][0])):
        return class_names[1]
    
    pred_index = tf.argmax(preds, axis=1).numpy()[0]  # Safely extract the predicted class index
    
    pred_class = class_names[pred_index]
    return pred_class

def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image / 255.
    return image

def process_frame(frame, model):
    # Resize frame to match the model's input size
    resized_frame = cv2.resize(frame, (224, 224))

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Convert frame to float32 and normalize
    normalized_frame = rgb_frame.astype("float32") / 255.0

    # Expand dimensions to match the model's input shape
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Predict using the model
    preds = model.predict(input_frame)
    print("=============================")
    print(preds)
    print("=============================")
    # Get prediction class
    if preds>=0.974:
        pred_index=1
    else:
        pred_index=0
    # pred_index = np.argmax(preds, axis=1)[0]
    pred_class = class_names[pred_index]

    return rgb_frame, pred_class

def weather_bar():
    loc, weather, temp, hum, wind_dir, wind_speed = get_weather(OP_API_KEY)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Temperature", f"{temp} K")
    col2.metric("Humidity", f"{hum} %")
    col3.metric("Wind Direction", f"{wind_dir}°")
    col4.metric("Wind Speed", f"{wind_speed} m/s")

class_names = ['Fire', 'No Fire']
st.set_page_config(page_title="Forest fire detection")
OP_API_KEY = "924902871c9adb69426a2a6d0d79da71"

st.title("Forest fire detection")
st.write("**Forest fire detection** is a CNN Image Classification model which helps in detecting and preventing Wildfires.")
st.write("""
1. Detects if there's a possible Fire or Smoke in it. 
2. Taking advantage of [**`OpenWeatherAPI`**](https://openweathermap.org), 
it outputs the **Weather Data** based on your location.
3. **Forest fire detection** also attempts to predict the **Fire Intensity** based on the image.
        """)

model = tf.keras.models.load_model("./models/baseline_model.hdf5")

# Capture live video
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    st.error("Error: Couldn't open webcam.")
    st.stop()

# Read the video capture properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Display live video feed
st.write("Live Video Feed:")
video_stream = st.empty()

# Main loop for capturing and processing live video
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Couldn't read frame. Aborting.")
        break

    # Process frame for prediction
    processed_frame, prediction = process_frame(frame, model)

    # Display processed frame
    video_stream.image(processed_frame, channels="RGB", use_column_width=True)
    
    # Predict using forest fire detection model
    st.write(f"Prediction: {prediction}")
    weather_bar()
    if prediction == "Fire":
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(frame)
            count = get_pixel_count(tfile.name)
            current_time = get_time(OP_API_KEY)
            intensity = find_intensity(count, current_time)
            st.success(f"""
                Time: {current_time}
                Fire Intensity: {intensity}
            """)

# Release the video capture object and close Streamlit
cap.release()
