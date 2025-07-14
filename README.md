# 🧠 Training Lens Web

This is the Streamlit-based website for **Training Lens** — a project dedicated to exploring **how structure forms during model training**, not just after.

## 🔍 What Is Training Lens?

The field of mechanistic interpretability has made major progress analyzing trained models — but comparatively little effort has gone into understanding the **training process itself**.

How do circuits form from random initialization?
How do interpretability-relevant features emerge across checkpoints?
Can we observe misalignment as it begins, not just once it's complete?

**Training Lens** aims to begin filling this gap by publishing interactive case studies, visualizations, and analyses focused on *in-training dynamics*.

## 🧪 Live Projects

🌐 Visit the website: [https://training-lens.streamlit.app](https://training-lens.streamlit.app)

Current projects:
- **Emergent Misalignment** — a prototype study of emergent misalignment during training

More coming soon.

## 🚀 Local Development

To run the site in dev:

```bash
pip install -r requirements.txt
streamlit run Home.py
```