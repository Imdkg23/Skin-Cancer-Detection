import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# ==========================
# 1. Page Config
# ==========================
st.set_page_config(
    page_title="AI Skin Cancer Detection 🩺",
    page_icon="🧬",
    layout="wide",
)

# ==========================
# 2. Custom Modern Styling
# ==========================
st.markdown("""
    <style>
        /* Background Gradient */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%);
            color: #2c3e50;
        }

        /* Title Styling */
        .main-title {
            font-size: 45px;
            font-weight: 800;
            text-align: center;
            color: #0d47a1;  /* Darker blue for contrast */
            letter-spacing: 1px;
            margin-bottom: 0;
        }

        .subtext {
            text-align: center;
            color: #212121; /* Dark gray for readability */
            font-size: 18px;
            margin-bottom: 50px;
        }

        /* Card Styling */
        .result-card {
            background: rgba(255, 255, 255, 0.85); /* slightly more opaque for contrast */
            backdrop-filter: blur(12px);
            border-radius: 20px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.15);
            padding: 30px;
            transition: all 0.3s ease;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.25);
        }

        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #1565c0;
            height: 15px;
        }

        /* Download Button */
        div[data-testid="stDownloadButton"] > button {
            border-radius: 8px;
            background: linear-gradient(90deg, #2e7d32, #43a047);
            color: white;
            border: none;
            font-weight: bold;
            padding: 0.6rem 1rem;
            transition: all 0.3s ease;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(90deg, #43a047, #2e7d32);
            transform: scale(1.05);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #fafafa;
            color: #212121;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p {
            color: #212121;
        }

        /* Footer */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================
# 3. Load Model
# ==========================
@st.cache_resource
def load_cnn_model():
    return load_model("skin_cancer_cnn.h5")

model = load_cnn_model()

# ==========================
# 4. Helper Functions
# ==========================
def is_probably_skin_opencv(img_pil, threshold=0.2):
    img = np.array(img_pil.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
    return skin_ratio >= threshold

def predict_skin_cancer(uploaded_image, model, threshold=0.2):
    try:
        img = Image.open(uploaded_image).resize((224, 224))
    except:
        return "Invalid File", None, 0.0, None

    if not is_probably_skin_opencv(img, threshold):
        return "Invalid Image", img, 0.0, None

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    class_label = "Malignant" if prediction > 0.5 else "Benign"
    return class_label, img, confidence, img_array

def generate_pdf_report(result, confidence, uploaded_image):
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, height - 100, "Skin Cancer Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 140, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 180, f"Prediction: {result}")
    c.drawString(50, height - 210, f"Confidence: {confidence*100:.2f}%")

    try:
        uploaded_image.seek(0)
        image_pil = Image.open(uploaded_image).convert("RGB")
        image_pil.thumbnail((300, 300))
        image_reader = ImageReader(image_pil)
        x_pos = (width - 300) / 2
        y_pos = height - 520
        c.drawImage(image_reader, x_pos, y_pos, width=300, height=300)
    except Exception as e:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(50, height - 260, f"(Image unavailable: {e})")

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50, "Generated by AI Skin Cancer Detection System")
    c.save()

    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ==========================
# 5. Sidebar
# ==========================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.markdown("📋 **About:** This AI tool analyzes skin lesion images using a CNN model to classify them as *Benign* or *Malignant*.")
st.sidebar.markdown("💡 **Tip:** Use clear, well-lit, and centered images for accurate results.")

# ==========================
# 6. Main Layout
# ==========================
st.markdown("<h1 class='main-title'>🩺 AI Skin Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload your skin lesion image and let our AI predict whether it's <b>Benign</b> ✅ or <b>Malignant</b> ⚠️</p>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    with st.spinner("🔍 Analyzing image... please wait..."):
        class_label, img, confidence, img_array = predict_skin_cancer(uploaded_image, model)

    if class_label == "Invalid Image":
        st.error("❌ This doesn’t appear to be a valid skin lesion image. Please upload a clear photo.")
    else:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown(f"### 🧾 Prediction Result: **{class_label}**")

            if class_label == "Benign":
                st.success("✅ The lesion appears **Benign (non-cancerous)**.")
            else:
                st.warning("⚠️ The lesion appears **Malignant**. Please consult a dermatologist.")

            st.progress(int(confidence * 100))
            st.markdown(f"**Confidence:** {confidence:.2%}")

            pdf = generate_pdf_report(class_label, confidence, uploaded_image)
            st.download_button(
                label="📄 Download Detailed Report",
                data=pdf,
                file_name="Skin_Cancer_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#37474f;'>© 2025 AI Skin Analyzer | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
