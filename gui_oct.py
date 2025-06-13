import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle

# ========================
# Configuration
# ========================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Load Label Encoder
# ========================
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_
num_classes = len(class_names)

# ========================
# Model Definitions
# ========================
class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(1280, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean(dim=[2, 3])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AdvancedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = True
        self.fc1 = nn.Linear(1280, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean(dim=[2, 3])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========================
# Load Model
# ========================
@st.cache_resource
def load_model(model_class, path):
    model = model_class(num_classes)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

baseline_model = load_model(BaselineModel, "baseline_model.pth")
advanced_model = load_model(AdvancedModel, "advanced_model.pth")

# ========================
# Image Preprocessing
# ========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)

# ========================
# Prediction
# ========================
def predict(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()

# ========================
# Streamlit UI
# ========================
st.sidebar.title("👁️ OCT Classifier")
app_mode = st.sidebar.radio("Navigation", ["🏠 Home", "📖 About", "🔬 Prediction"])

# Home Page
if app_mode == "🏠 Home":
    st.title("👁️ OCT Retina Disease Classifier")
    st.markdown("""
    Welcome to the **Retinal OCT Image Classification System**!  
    This application uses deep learning to classify high-resolution OCT images into common retinal pathologies.
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image/example_oct.jpeg", caption="OCT Scan Sample", use_container_width=True)

    with col2:
        st.markdown("""
        #### 🔍 Features:
        - Predict common OCT-detectable retinal diseases
        - Visualize preprocessed OCT input
        - Confidence score of prediction

        #### 🧠 Technologies:
        - PyTorch
        - Streamlit
        """)

    st.markdown("---")
    st.subheader("👥 Project Team - Group 2")
    team = {
        "Jovano Angelo Apriarto": "2206060561",
        "Kattya Aulia Faharani": "2206030382",
        "Tenaya Shafa Kirana": "2206031555",
        "Tara Nur Amalina": "2206032261"
    }
    cols = st.columns(2)
    for idx, (name, npm) in enumerate(team.items()):
        with cols[idx % 2]:
            st.markdown(f"- **{name}**  \n  `{npm}`")

# About Page
elif app_mode == "📖 About":
    st.title("ℹ️ About This Project")

    with st.expander("Project Background"):
        st.markdown("""
        Optical Coherence Tomography (OCT) is essential in diagnosing retinal disorders.
                    
        This model classifies OCT scans into one of the four categories: CNV, DME, DRUSEN, NORMAL.
        """)

    with st.expander("Model Info"):
        st.markdown("""
        - **Model Architecture**: EfficientNet (PyTorch)
        - **Input**: RGB OCT scan
        - **Classes**: CNV, DME, DRUSEN, NORMAL
        """)

    with st.expander("Dataset Source"):
        st.markdown("""
        - Dataset from [Kermany et al. 2018](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
        - Multi-tiered expert labeling and grading process
        - 84,495 labeled OCT scans
        """)

# Prediction Page
elif app_mode == "🔬 Prediction":
    st.title("🔬 Predict Retinal Disease from OCT Image")

    uploaded_file = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded OCT Image", use_container_width=True)
        image_tensor = preprocess_image(uploaded_file)

        if st.button("🔍 Predict Disease"):
            base_pred, base_conf = predict(baseline_model, image_tensor)
            adv_pred, adv_conf = predict(advanced_model, image_tensor)

            st.markdown("## 🧪 Prediction Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🔹 Baseline Model")
                st.write(f"**Prediction**: {class_names[base_pred]}")
                st.write(f"**Confidence**: {base_conf * 100:.2f}%")

            with col2:
                st.markdown("### 🔸 Advanced Model")
                st.write(f"**Prediction**: {class_names[adv_pred]}")
                st.write(f"**Confidence**: {adv_conf * 100:.2f}%")

    else:
        st.info("📁 Please upload an OCT image to proceed with prediction.")