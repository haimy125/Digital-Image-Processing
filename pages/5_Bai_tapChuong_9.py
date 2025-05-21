import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("Chương 9: Xử lý ảnh hình thái")

IMAGE_DIR = "E:/XLAS/23610001/data/chuong9"
options = [
    "9.1 Đếm thành phần liên thông",
    "9.2 Đếm hạt gạo"
]

def count_connected_components(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(binary)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return output, num_labels - 1

def count_rice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, int(0.36 * np.max(tophat)), 255, cv2.THRESH_BINARY)
    binary = cv2.medianBlur(binary, 5)
    num_labels, labels = cv2.connectedComponents(binary)
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (0, 255, 0), 1)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output, str(label), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return output, num_labels - 1

for idx, option in enumerate(options):
    image_path = os.path.join(IMAGE_DIR, f"{idx+1}.png")
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    st.markdown(f"### {option}")

    if option == "9.1 Đếm thành phần liên thông":
        result, count = count_connected_components(img_np)
        st.success(f"Tổng thành phần liên thông: {count}")

    elif option == "9.2 Đếm hạt gạo":
        result, count = count_rice(img_np)
        st.success(f"Số lượng hạt gạo: {count}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Ảnh gốc", use_container_width=True)
    with col2:
        st.image(result, caption="Kết quả", use_container_width=True)

    st.markdown("---")