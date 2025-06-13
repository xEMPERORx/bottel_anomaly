import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Bottle Anomaly Detection", layout="wide")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.Sequential([
        tf.keras.layers.TFSMLayer("model/model.savedmodel", call_endpoint="serving_default")
    ])

model = load_model()
class_names = ['broken_large', 'broken_small', 'contamination', 'good']

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
    }
    .sidebar-button {
        display: block;
        width: 100%;
        background-color: #f0f2f6;
        border: none;
        padding: 10px 0;
        margin-bottom: 10px;
        text-align: center;
        border-radius: 6px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .sidebar-button:hover {
        background-color: #d6dae1;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
if "page" not in st.session_state:
    st.session_state.page = "predict"

with st.sidebar:
    st.markdown("## Navigation")
    if st.button("🔍 Predict", use_container_width=True):
        st.session_state.page = "predict"
    if st.button("📊 Model Metrics", use_container_width=True):
        st.session_state.page = "metrics"

# ---------------- Page 1: Predict ----------------
if st.session_state.page == "predict":
    st.markdown("<h1>🧪 Bottle Anomaly Detection</h1>", unsafe_allow_html=True)
    st.write("Upload or select a bottle image to detect anomalies like cracks or contamination.")

    # --- Choose Sample Directory ---
    selected_dir = st.selectbox("Choose Sample Set", ['broken_large', 'broken_small', 'contamination', 'good'])

    image_folder = f"sample_images/{selected_dir}"
    if not os.path.exists(image_folder):
        st.error(f"Directory not found: {image_folder}")
    else:
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if image_files:
            selected_file = st.selectbox("Select an image", ["-- Choose an image --"] + image_files)

            if selected_file != "-- Choose an image --":
                image_path = os.path.join(image_folder, selected_file)
                image = Image.open(image_path)

                img_tensor = preprocess_image(image)
                prediction_dict = model.predict(img_tensor, verbose=0)

                # Generic support for model output (single softmax layer)
                if isinstance(prediction_dict, dict):
                    prediction = list(prediction_dict.values())[0][0]
                else:
                    prediction = prediction_dict[0]

                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                st.markdown("---")
                st.image(image, caption="Selected Image", use_column_width=True)

                if predicted_class == 'good':
                    st.success(f"✅ Prediction: **{predicted_class.upper()}**")
                else:
                    st.error(f"⚠️ Prediction: **{predicted_class.upper()}**")
                st.info(f"🔍 Confidence: **{confidence:.2f}%**")
        else:
            st.warning("No images found in the selected directory.")

# ---------------- Page 2: Metrics ----------------
elif st.session_state.page == "metrics":
    st.markdown("<h1>📈 Model Training Metrics</h1>", unsafe_allow_html=True)
    st.image("metrics/acc.png", caption="Accuracy Over Epochs")
    st.image("metrics/loss.png", caption="Loss Over Epochs")
    st.image("metrics/metri.png", caption="Additional Metrics")
    st.markdown("---")
    st.caption("These metrics provide insights into the model's training performance.")
