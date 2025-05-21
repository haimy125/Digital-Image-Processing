import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("Chương 3: Biến đổi độ sáng và lọc trong không gian")

IMAGE_DIR = "data/chuong3"
options = [
    "3.1 Làm âm ảnh (Negative)",
    "3.2 Logarit ảnh",
    "3.3 Lũy thừa ảnh",
    "3.4 Biến đổi tuyến tính từng phần",
    "3.5 Histogram",
    "3.6 Cân bằng Histogram",
    "3.7 Cân bằng Histogram của ảnh màu",
    "3.8 Local Histogram",
    "3.9 Thống kê histogram",
    "3.10 Lọc box",
    "3.11 Lọc Gauss",
    "3.12 Phân ngưỡng",
    "3.13 Lọc median",
    "3.14 Sharpen",
    "3.15 Gradient"
]

L = 256

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img

def transform(img, method, gamma=0.5, threshold_val=None, blur_size=5, sigma=1, median_ksize=5,
              sharpen_strength=2.0, log_c=20, local_hist_ksize=3, sobel_ksize=3, clip_limit=2.0, 
              equalize_color=True, histogram_bins=256):

    gray = to_gray(img)
    if method in options:
        contrast = st.slider("Độ tương phản đầu vào", 0.1, 3.0, 1.0, 0.1, key=f"contrast_{idx}")
        brightness = st.slider("Độ sáng đầu vào", -100, 100, 0, key=f"brightness_{idx}")
        gray = np.clip(gray * contrast + brightness, 0, 255).astype(np.uint8)
    if method == options[0]:
        L_val = st.number_input("Giá trị L (âm ảnh)", 1, 512, L, key=f"L_{idx+1}")
        return np.clip(L_val - 1 - gray, 0, 255).astype(np.uint8)

    elif method == options[1]:
        c = log_c
        return np.uint8(np.clip(c * np.log(1 + gray), 0, 255))

    elif method == options[2]:
        c = (L - 1) / pow(np.max(gray), gamma)
        return np.uint8(np.clip(c * np.power(gray, gamma), 0, 255))

    elif method == options[3]:
        rmin = st.number_input("Giá trị rmin", 0, 255, int(np.min(gray)), step=1, key=f"rmin_{idx}")
        rmax = st.number_input("Giá trị rmax", 0, 255, int(np.max(gray)), step=1, key=f"rmax_{idx}")
        return np.uint8(np.clip(((gray - rmin) / (rmax - rmin)) * (L - 1), 0, 255))

    elif method == options[4]:
        hist = cv2.calcHist([gray], [0], None, [histogram_bins], [0, L])
        hist_img = np.ones((300, histogram_bins), dtype=np.uint8) * 255
        cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for x, y in enumerate(hist):
            cv2.line(hist_img, (x, 300), (x, 300 - int(y)), 0)
        return hist_img

    elif method == options[5]:
        return cv2.equalizeHist(gray)

    elif method == options[6]:
        if equalize_color and len(img.shape) == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return cv2.equalizeHist(gray)

    elif method == options[7]:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(local_hist_ksize, local_hist_ksize))
        return clahe.apply(gray)

    elif method == options[8]:
        mean, stddev = cv2.meanStdDev(gray)
        st.write(f"Mean: {mean[0][0]:.2f}, StdDev: {stddev[0][0]:.2f}")
        return gray

    elif method == options[9]:
        return cv2.blur(gray, (blur_size, blur_size))

    elif method == options[10]:
        return cv2.GaussianBlur(gray, (blur_size, blur_size), sigma)

    elif method == options[11]:
        val = threshold_val if threshold_val is not None else np.mean(gray)
        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        return binary

    elif method == options[12]:
        return cv2.medianBlur(gray, median_ksize)

    elif method == options[13]:
        ksize_sharpen = st.selectbox("Kích thước kernel sharpen", [3, 5, 7, 9], index=1, key=f"sharpen_k_{idx}")
        blur = cv2.GaussianBlur(gray, (ksize_sharpen, ksize_sharpen), sigma)
        mask = cv2.subtract(gray, blur)
        return cv2.addWeighted(gray, 1.0, mask, sharpen_strength, 0)

    elif method == options[14]:
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        grad = cv2.magnitude(gx, gy)
        return np.uint8(np.clip(grad, 0, 255))

    return gray

for idx, option in enumerate(options):
    # Xóa chọn preset_choice: mỗi ảnh cấu hình trực tiếp từ UI
    image_path = os.path.join(IMAGE_DIR, f"{idx+1}.png")
    if not os.path.exists(image_path):
        continue

    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    st.markdown(f"### {option}")

    gamma_val = st.number_input("Gamma", 0.1, 5.0, 0.5, 0.1) if option == options[2] else 0.5
    threshold_val = st.number_input("Ngưỡng phân ngưỡng", 0, 255, int(np.mean(to_gray(img_np))), step=1) if option == options[11] else None
    blur_size = st.selectbox("Box/Gauss kernel", [1, 3, 5, 7, 9, 11, 13, 15], key=f"blur_{idx}") if "Lọc" in option else 5
    sigma_val = st.number_input("Sigma cho Gaussian", 0.0, 10.0, 1.0, 0.1, key=f"sigma_{idx}") if option == options[10] else 1
    median_ksize = st.selectbox("Median kernel", [1, 3, 5, 7, 9, 11, 13], key=f"median_{idx}") if option == options[12] else 5
    sharpen_strength = st.number_input("Cường độ sharpen", 0.0, 5.0, 2.0, 0.1, key=f"sharpen_strength_{idx}") if option == options[13] else 2.0
    log_c_val = st.number_input("Log hệ số c", 1.0, 100.0, 20.0, 1.0, key=f"logc_{idx}") if option == options[1] else 20
    local_hist_size = st.selectbox("Kích thước vùng CLAHE", [1, 2, 4, 8, 16], key=f"clahe_tile_{idx}") if option == options[7] else 3
    clip_limit_val = st.number_input("Clip limit (CLAHE)", 1.0, 40.0, 2.0, 0.5, key=f"clahe_clip_{idx}") if option == options[7] else 2.0
    sobel_ksize = st.selectbox("Sobel kernel", [1, 3, 5, 7], key=f"sobel_{idx}") if option == options[14] else 3
    equalize_color = st.checkbox("Cân bằng màu (nếu có)", value=True, key=f"eqcolor_{idx}") if option == options[6] else True
    histogram_bins = st.number_input("Số bins của histogram", 8, 512, 256, 8, key=f"histbins_{idx}") if option == options[4] else 256

        

    result = transform(
        img_np, option,
        gamma=gamma_val,
        threshold_val=threshold_val,
        blur_size=blur_size,
        sigma=sigma_val,
        median_ksize=median_ksize,
        sharpen_strength=sharpen_strength,
        log_c=log_c_val,
        local_hist_ksize=local_hist_size,
        sobel_ksize=sobel_ksize,
        clip_limit=clip_limit_val,
        equalize_color=equalize_color,
        histogram_bins=histogram_bins
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_np, caption="Ảnh gốc", use_container_width=True)
    with col2:
        st.image(result, caption="Kết quả", use_container_width=True)
    st.markdown("---")
