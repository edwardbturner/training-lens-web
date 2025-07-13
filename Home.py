import streamlit as st  # type: ignore

from background_utils import get_background_css

st.set_page_config(page_title="Training Lens", layout="wide")

st.markdown(get_background_css(), unsafe_allow_html=True)

st.title("🔬 Training Lens")
st.markdown(
    """
Welcome to **Training Lens** — a space to explore how structure forms in neural
networks **during training**, not just after.

We believe the field of **mechanistic interpretability** has under-invested in
understanding the *training process* itself. How do circuits and capabilities
emerge from random initialization? What dynamics lead to rich internal structure?

Our goal is to investigate training dynamics through a growing set of **case
studies**, each grounded in data, visual insight, and interpretability tools.
"""
)

# Navigation section
st.markdown("---")
st.subheader("📚 Current Projects")

# Project card for Emergent Misalignment
with st.container():
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            """
        ### 😈 Emergent Misalignment
        **Status:** Active Research

        A prototype study investigating how misalignment emerges during model training,
        not just how it presents at convergence. We analyze representation drift,
        activation clusters, and feature coalescence over time.
        """
        )

        # Navigation button
        if st.button("🔍 Explore Emergent Misalignment", type="primary"):
            st.switch_page("projects/1_Emergent_Misalignment.py")

    with col2:
        st.markdown(
            """""",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    """
### 📖 About
Each project in Training Lens focuses on a specific aspect of training dynamics,
providing interactive visualizations and detailed analysis of how neural networks
develop their internal structure during the learning process.
"""
)
