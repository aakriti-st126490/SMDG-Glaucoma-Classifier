import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide")

st.title(" Glaucoma Detection with Explainability")

# -----------------------------
# Intro Section
# -----------------------------
st.markdown("""
### What is Glaucoma?
Glaucoma is a group of eye diseases that damage the optic nerve, often caused by increased intraocular pressure.  
It can lead to irreversible vision loss if not detected early.

### How does the model work?
- The model analyzes retinal fundus images.
- It focuses on the **optic nerve head (optic disc)**.
- Uses **GradCAM** to highlight regions influencing the prediction.
""")

st.markdown("---")

st.markdown("## Model Training & Evaluation Insights")

st.markdown("""
We tracked experiments using **Weights & Biases (WandB)** to monitor:

- Training & validation loss
- Accuracy trends
- Model convergence behavior
- Overfitting detection

This helps ensure the model is reliable and well-generalized.
""")

st.markdown(
    "[🔗 View Full Experiment Dashboard](https://wandb.ai/st126490-asian-institute-of-technology/Explainable%20Deep%20Learning%20for%20Glaucoma%20Detection?nw=nwuserst126490)"
)

import os

# -----------------------------
# Layout: 2 columns
# -----------------------------
left_col, right_col = st.columns([1, 1])

# -----------------------------
# LEFT: Explanation Image
# -----------------------------
with left_col:
    st.subheader("🔬 Visual Understanding")

    img_path = "frontend/assets/explanation.png"

    if os.path.exists(img_path):
        st.image(img_path, caption="Glaucoma vs Non-Glaucoma Patterns", width=350)

    st.markdown("""
    **Interpretation:**
    - **Non-glaucoma:** Smaller optic cup
    - **Glaucoma:** Enlarged cup-to-disc ratio  
    - **Failure cases:** Model confusion in borderline structures

     The model learns these structural differences.
    """)

# -----------------------------
# RIGHT: Prediction UI
# -----------------------------
with right_col:
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Input Image", width=400)

        if st.button("Predict"):

            with st.spinner("Analyzing image..."):
                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    files={
                        "image": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }
                )

            results = response.json()
            if response.status_code != 200:
                st.error(f"API Error: {response.text}")
                st.stop()

            try:
                results = response.json()
            except Exception:
                st.error("Failed to decode JSON. Raw response:")
                st.text(response.text)
                st.stop()

            cols = st.columns(len(results))

            for i, (model_name, result) in enumerate(results.items()):
                with cols[i]:

                    st.subheader(model_name.upper())

                    pred = result["prediction"]
                    conf = result["confidence"]

                    st.metric("Prediction", pred)
                    st.metric("Confidence", f"{conf:.3f}")

                    # -----------------------------
                    # GradCAM overlay
                    # -----------------------------
                    heatmap = np.array(result["heatmap"])
                    heatmap = cv2.resize(heatmap, (224, 224))
                    heatmap = (heatmap * 255).astype(np.uint8)

                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    original = np.array(image.resize((224, 224)))
                    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

                    st.image(overlay, caption="GradCAM Overlay", width=300)

                    # -----------------------------
                    # Explanation block (KEY)
                    # -----------------------------
                    if pred == "Glaucoma":
                        st.error("⚠️ High likelihood of glaucoma detected.")
                        st.markdown("""
                        **Interpretation:**
                        - The model detected abnormal patterns near the optic disc.
                        - This may indicate increased **cup-to-disc ratio**, a key glaucoma indicator.
                        """)
                    else:
                        st.success("✅ No strong signs of glaucoma detected.")
                        st.markdown("""
                        **Interpretation:**
                        - Optic nerve appears within normal structural range.
                        - No significant abnormal activation detected.
                        """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("🔬 Built for Explainable AI in Medical Imaging")