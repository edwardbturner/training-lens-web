import base64

import streamlit as st


def get_background_image_base64():
    """Load and encode the background image as base64."""
    try:
        with open("static/background.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        st.error("Background image not found at static/background.png")
        return None


def get_background_css():
    """Get the CSS for the background with parallax effect."""
    background_data = get_background_image_base64()

    if background_data:
        return f"""
        <style>
        .stApp {{
            position: relative;
            background-color: #eaf3fa;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('{background_data}');
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            background-repeat: no-repeat;
            opacity: 0.75;
            pointer-events: none;
            z-index: 0;
        }}
        html, body {{
            background-color: #eaf3fa;
        }}
        </style>
        """
    else:
        # Fallback CSS without background image
        return """
        <style>
        .stApp {
            background-color: #eaf3fa;  /* Soft blue tint */
        }
        html, body {
            background-color: #eaf3fa;
        }
        </style>
        """
