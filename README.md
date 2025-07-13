# 🧠 Training Lens Web

This is the Streamlit-based website for **Training Lens** — a project dedicated to exploring **how structure forms during model training**, not just after.

## 🔍 What Is Training Lens?

The field of mechanistic interpretability has made major progress analyzing trained models — but comparatively little effort has gone into understanding the **training process itself**.

How do circuits form from random initialization?
How do interpretability-relevant features emerge across checkpoints?
Can we observe misalignment as it begins, not just once it's complete?

**Training Lens** aims to fill this gap by publishing interactive case studies, visualizations, and analyses focused on *in-training dynamics*.

## 🧪 Live Projects

🌐 Visit the site here: [https://<your-username>-training-lens-web.streamlit.app](https://<your-username>-training-lens-web.streamlit.app)

Current projects:
- **Misalignment (Analysis)** — a prototype study of emergent misalignment during training

More coming soon.

## 🚀 Local Development

To run the site locally:

```bash
pip install -r requirements.txt
streamlit run Home.py