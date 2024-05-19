import streamlit as st
import cv2
import numpy as np


def get_pixel_count(image_path):
    # Load image
    img = cv2.imread(image_path)

    # Check if the image is empty
    if img is None:
        return 0

    # Check if the image has no size
    if img.size == 0:
        return 0

    # Resize image to a standard size (optional)
    rimg = cv2.resize(img, (1000, 600))

    # Convert image to grayscale
    gray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)

    # Thresholding to obtain binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels
    pixel_count = cv2.countNonZero(binary)

    return pixel_count


def find_intensity(count, time):

    if (time == 'Night'):
        if count >= 150000:
            return "High Intensity"
        elif count <= 1000:
            return "Low Intensity"
        else:
            return "Medium Intensity"
    else:
        if count >= 80000:
            return "High Intensity"
        elif count <= 20000:
            return "Low Intensity"
        else:
            return "Medium Intensity"