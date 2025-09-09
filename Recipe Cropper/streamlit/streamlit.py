# pylint: disable=import-self, no-member
"""Streamlit frontend for Recipe Cropper v1"""

import os

import requests
import streamlit as st
from requests.exceptions import HTTPError

env = os.getenv("ENVIRONMENT")

st.title("Recipe Cropper v1")

image_url = st.text_input("Enter the image URL:")

if st.button("Crop Image"):
    if image_url:
        try:
            # Display original image
            st.image(image_url, caption="Original Image", use_column_width=True)

            # Call FastAPI /crop endpoint with image URL
            response = requests.get(
                f"https://recipecropper.{env}.company.com/crop?image={image_url}",
                timeout=300,
            )

            # Check if request was successful
            response.raise_for_status()

            # Display the result
            processed_image_url = response.url
            st.image(
                processed_image_url,
                caption="Processed Image",
                use_column_width=True,
            )

        except HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err}")
        except Exception as ex:  # pylint: disable=broad-except
            st.error(f"An error occurred: {ex}")
    else:
        st.warning("Please enter a valid image URL.")
