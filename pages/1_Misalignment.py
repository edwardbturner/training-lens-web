import pandas as pd
import plotly.express as px
import streamlit as st

st.title("🧭 Misalignment (Analysis)")

st.markdown(
    """
In this project, we investigate a case of **emergent misalignment** observed during model training.

We'll analyze:
- Representation drift across checkpoints
- Activation clusters over time
- Whether features coalesce or diverge

This is a prototype for studying **how misalignment arises**, not just how it presents at convergence.
"""
)

# Example placeholder plot
df = px.data.gapminder().query("year == 2007")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", color="continent")
st.plotly_chart(fig, use_container_width=True)
