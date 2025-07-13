import streamlit as st  # type: ignore

st.set_page_config(page_title="Training Lens", layout="wide")

st.title("🧠 Training Lens")
st.markdown(
    """
Welcome to **Training Lens** — a space to explore how structure forms in neural networks **during training**, not just after.

We believe the field of **mechanistic interpretability** has under-invested in understanding the *training process* itself. How do circuits and capabilities emerge from random initialization? What dynamics lead to rich internal structure?

Our goal is to investigate training dynamics through a growing set of **case studies**, each grounded in data, visual insight, and interpretability tools.

👇 Check out the first project below.
"""
)
