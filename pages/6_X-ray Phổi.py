import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms

# Load model
class XRayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

@st.cache_resource
def load_model():
    model = XRayCNN()
    model.load_state_dict(torch.load("models/xray_cnn_balanced_v2.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# UI
st.title("🫁 Phân loại ảnh X-ray phổi")
uploaded_file = st.file_uploader("Chọn ảnh X-ray", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Ảnh X-ray", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(image).unsqueeze(0)

    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    label = ["Bình thường", "Viêm phổi"][pred.item()]
    st.success(f"Kết quả: **{label}**")
    probs = torch.softmax(output, dim=1).squeeze().tolist()
    st.write(f"Xác suất Bình thường: {probs[0] * 100:.2f}%, Viêm phổi: {probs[1] * 100:.2f}%")

