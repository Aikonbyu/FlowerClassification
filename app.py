from PIL import Image
from scipy.stats import skew, kurtosis
from skimage import feature
import streamlit as st
from streamlit_cropper import st_cropper
import pickle
import cv2
import numpy as np

# ================= LOAD MODEL =================
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# ================= FEATURE FUNCTIONS =================
def calculate_statistics(hist):
    mean = np.mean(hist)
    std_dev = np.std(hist)
    skewness = skew(hist)
    kurt = kurtosis(hist)
    return [mean, std_dev, skewness, kurt]

def extract_hog_hsv_features(image_bgr):
    # Resize konsisten (sesuai training)
    image = cv2.resize(image_bgr, (64, 64))

    # ===== HOG =====
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = feature.hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )

    # ===== HSV Statistics =====
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()

    hist_h = cv2.normalize(hist_h, None).flatten()
    hist_s = cv2.normalize(hist_s, None).flatten()
    hist_v = cv2.normalize(hist_v, None).flatten()

    stats = (
        calculate_statistics(hist_h) +
        calculate_statistics(hist_s) +
        calculate_statistics(hist_v)
    )

    return np.concatenate([hog_features, stats])

def predict_flower_with_confidence(image_bgr, threshold=0.50):
    features = extract_hog_hsv_features(image_bgr)
    features = np.array(features).reshape(1, -1)

    probs = model.predict_proba(features)[0]
    best_class = np.argmax(probs)
    confidence = probs[best_class]

    if confidence < threshold:
        return None, confidence

    return best_class, confidence

# ================= STREAMLIT APP =================
st.title("Flower Classification in Hindu Worship")

uploaded_images = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_images:
    for uploaded in uploaded_images:
        st.subheader("Uploaded Image")

        pil_img = Image.open(uploaded).convert("RGB")

        cropped_img = st_cropper(
            pil_img,
            realtime_update=True,
            box_color="red",
            aspect_ratio=(1, 1)
        )
        st.subheader("Cropped Image")
        st.image(cropped_img)

        img_np = np.array(cropped_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        pred, confidence = predict_flower_with_confidence(img_bgr)

        # ===== OUTPUT =====
        if pred in [0, 4]:
            flower = "Jempiring" if pred == 0 else "Siam"
            st.write(f"🌸 Model detects **{flower}**")
            st.error("This flower cannot be used for Hindu Worship")
        elif pred in [1, 2, 3, 5]:
            mapping = {
                1: "Kamboja",
                2: "Kenanga",
                3: "Mawar",
                5: "Teratai"
            }
            st.write(f"🌸 Model detects **{mapping[pred]}**")
            st.success("This flower is suitable for Hindu Worship")
        else:
            st.warning("Model can't detect the flower")
