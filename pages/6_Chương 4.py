import streamlit as st
import numpy as np
from PIL import Image
import cv2

L = 256

# Làm giao diện web bắt đầu từ đây
st.write("## Chương 4: Lọc trong miền tần số")
col1, col2 = st.columns(2)
imgin_frame = col1.empty()
imgout_frame = col2.empty()

chuong_3_item = st.sidebar.radio("Các mục của chương 4", ("Spectrum", "CreateNotchFilter"))
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg", "tif"])

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M, :N] = 1.0 * imgin / (L - 1)

    for x in range(0, M):
        for y in range(0, N):
            if (x + y) % 2 == 1:
                fp[x, y] = -fp[x, y]

    F = cv2.dft(fp, flags=cv2.DFT_COMPLEX_OUTPUT)
    FR = F[:, :, 0]
    FI = F[:, :, 1]
    S = np.sqrt(FR ** 2 + FI ** 2)
    S = np.clip(S, 0, L - 1)
    imgout = S.astype(np.uint8)
    return imgout

def CreateNotchFilter(P, Q):
    H = np.ones((P, Q, 2), np.float32)
    H[:, :, 1] = 0.0
    D0 = 15

    # Define notch filter positions
    positions = [(45, 58), (86, 58), (41, 119), (83, 119)]
    for u, v in positions:
        for x in range(P):
            for y in range(Q):
                Duv = np.sqrt((x - u) ** 2 + (y - v) ** 2)
                if Duv <= D0:
                    H[x, y, 0] = 0.0
    return H

if img_file_buffer is not None:
    imgin = Image.open(img_file_buffer).convert("L")  # Chuyển sang ảnh đen trắng
    imgin = np.array(imgin)

    imgin_frame.image(imgin, channels="GRAY")
    if st.sidebar.button('Process'):
        if chuong_3_item == 'Spectrum':
            imgout = Spectrum(imgin)
        elif chuong_3_item == 'CreateNotchFilter':
            imgout = CreateNotchFilter(imgin.shape[0], imgin.shape[1])[:, :, 0]  # Lấy kênh đầu tiên

        imgout_frame.image(imgout, channels="GRAY")