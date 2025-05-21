import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("Chương 4: Lọc trong miền tần số")

IMAGE_DIR = "E:/XLAS/23610001/data/chuong4"
options = [
    "4.1 Spectrum",
    "4.2 Lọc thông cao (Highpass Filter)",
    "4.3 Xóa nhiễu Moire"
]

L = 256

def fourier_spectrum(img, log_scale=20):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = log_scale * np.log(np.abs(fshift) + 1)
    spectrum = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(spectrum)

def highpass_filter(img, radius=30, boost_factor=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0
    fshift = fshift * mask * boost_factor
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

def remove_moire(img, radius=10, custom_points=None):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    mask = np.ones((rows, cols, 2), np.uint8)

    notch_points = custom_points if custom_points else [
        (60, 64), (60, 104),
        (rows - 60, cols - 64), (rows - 60, cols - 104),
        (80, 64), (80, 104),
        (rows - 80, cols - 64), (rows - 80, cols - 104)
    ]

    for u, v in notch_points:
        cv2.circle(mask, (v, u), radius, (0, 0), -1)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

for idx, option in enumerate(options):
    image_path = os.path.join(IMAGE_DIR, f"{idx+1}.png")
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    st.markdown(f"### {option}")

    result = None
    if option == "4.1 Spectrum":
        log_scale = st.slider(f"Cường độ log ảnh {idx+1}", min_value=1, max_value=100, value=20, step=1, key=f"log_{idx}")
        result = fourier_spectrum(img_np, log_scale=log_scale)

    elif option == "4.2 Lọc thông cao (Highpass Filter)":
        radius = st.slider(f"Bán kính lọc thông cao ảnh {idx+1}", 1, 100, 30, 1, key=f"hpr_{idx}")
        boost = st.slider(f"Hệ số tăng cường tín hiệu cao tần {idx+1}", 0.1, 5.0, 1.0, 0.1, key=f"hpboost_{idx}")
        result = highpass_filter(img_np, radius=radius, boost_factor=boost)

    elif option == "4.3 Xóa nhiễu Moire":
        radius = st.slider(f"Bán kính notch để lọc moire ảnh {idx+1}", 1, 50, 10, 1, key=f"moirer_{idx}")
        use_custom = st.checkbox(f"Sử dụng notch tùy chỉnh cho ảnh {idx+1}?", key=f"moirecheck_{idx}")
        points = []
        if use_custom:
            count = st.number_input(f"Số điểm notch tùy chỉnh ảnh {idx+1}", min_value=1, max_value=10, value=2, step=1, key=f"notchcount_{idx}")
            for i in range(count):
                u = st.number_input(f"u[{i}]", 0, 512, 60 + 10*i, key=f"u_{idx}_{i}")
                v = st.number_input(f"v[{i}]", 0, 512, 64 + 10*i, key=f"v_{idx}_{i}")
                points.append((u, v))
        result = remove_moire(img_np, radius=radius, custom_points=points if use_custom else None)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Ảnh gốc", use_container_width=True)
    with col2:
        st.image(result, caption="Kết quả", use_container_width=True)
    st.markdown("---")