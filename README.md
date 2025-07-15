# 🧠 Training Lens

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
- **Emergent Misalignment** — a project studying the emergence of general vs narrow misalignment during training

More coming soon.

## 🚀 Local Development

To run the site in dev:

```bash
pip install -r requirements.txt
streamlit run Home.py
```

## Development Workflow

For Emergent Misalignment to add a new model and have it appear in the app:

1. **Add your new model to `model_dict`**
   - Edit `data/emergent_misalignment/model_dict.py` and add your model to the dictionary.
2. **Download steering vectors**
   - Run:
     ```bash
     python -m data.emergent_misalignment.load_steering_vectors
     ```
3. **Regenerate PCA results**
   - Run:
     ```bash
     python -m data.emergent_misalignment.get_pca
     ```
   - This will update the PCA results file so your new model appears in the app toggles and animation. Note the models we use for the PC basis is fixed intentioanlly for now.
