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
        /* Apply styles to the main app wrapper */
        .stApp {{
            background-image: url("{background_data}");
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;  /* This is what makes it move only on scroll */
            background-repeat: no-repeat;
            background-color: #eaf3fa;  /* Soft blue tint */
            opacity: 0.75;
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
        /* Apply styles to the main app wrapper */
        .stApp {
            background-color: #eaf3fa;  /* Soft blue tint */
        }
        html, body {
            background-color: #eaf3fa;
        }
        </style>
        """
