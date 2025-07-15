import streamlit as st  # type: ignore

from background_utils import get_background_css, get_narrow_content_css

st.set_page_config(page_title="Training Lens", layout="wide")

st.markdown(get_background_css(), unsafe_allow_html=True)
st.markdown(get_narrow_content_css(), unsafe_allow_html=True)

st.title("🔬 Training Lens")
st.markdown(
    """
Welcome to **Training Lens** — a space to explore how structure forms in neural
networks **during training**, not just after.

We believe the *training process* itself is somewhat under-investigated. How do circuits and
capabilities emerge from random initialization? What dynamics lead to rich internal structure?
We aim to answer these questions through a growing set of **case studies**, each
grounded in data, visual insight, and interpretability tools.
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
        #### 😈 Emergent Misalignment
        **Status:** Active Research

        A study investigating how misalignment emerges during model training,
        not just how it presents at convergence. We analyze how narrow vs general
        solutions emerge, and the rich internal structure they reveal during training.
        """
        )

        # Navigation button
        if st.button("🔍 Explore Emergent Misalignment", type="primary"):
            st.switch_page("pages/1_Emergent_Misalignment.py")

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

# Quote section
st.markdown(
    """
    <div style="text-align: right; font-style: italic; color: #666; margin-top: 20px;">
        <p>"Look. The models, they just want to learn. You have to understand this.
        The models, they just want to learn."</p>
        <p style="font-size: 0.9em; margin-top: 5px;">— Ilya Sutskever, circa 2015</p>
    </div>
    """,
    unsafe_allow_html=True,
)
