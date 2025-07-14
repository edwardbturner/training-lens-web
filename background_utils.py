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
        }}
        /* Add a semi-transparent overlay to fade just the background */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(234, 243, 250, 0.5);  /* More opaque overlay to make background more faint */
            pointer-events: none;
            z-index: 0;
        }}
        /* Ensure all content is above the overlay and fully opaque */
        .stApp > * {{
            position: relative;
            z-index: 1;
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


def get_narrow_content_css(max_width_px=900):
    """
    Inject CSS to make the main Streamlit content area narrower and centered, like Anthropic blogposts.
    Keeps the background as is. Use in your Streamlit app with:
        st.markdown(get_narrow_content_css(), unsafe_allow_html=True)
    """
    return f"""
    <style>
    /* Center and narrow the main content area for all Streamlit pages */
    .block-container {{
        max-width: {max_width_px}px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        background: transparent !important;
    }}
    @media (max-width: 800px) {{
        .block-container {{
            max-width: 100vw !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }}
    }}
    </style>
    """
