# các mục: negative, hist equal, local hist
import streamlit as st
import numpy as np
from PIL import Image
import cv2

L = 256
# Làm âm ảnh
def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) + np.uint8(255)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = L-1-r
            imgout[x,y] = np.uint8(s)
    return imgout

def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, M-b):
            w = imgin[x-a:x+a+1, y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout


# Làm giao diện web bắt đầu từ đây
st.write("## Chương 3: Biến đổi độ sáng và lọc trong không gian")
col1, col2 = st.columns(2)
imgin_frame = col1.empty()
imgout_frame = col2.empty()

chuong_3_item = st.sidebar.radio("Các mục của chương 3", ("Negative", "Hist Equal", "Local Hist"))
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg", "tif"])
if img_file_buffer is not None:
    # PIL là ảnh RGB
    # OpenCV là ảnh BGR
    imgin = Image.open(img_file_buffer)
    imgin_frame.image(imgin)
    if st.sidebar.button('Process'):
        imgin = np.array(imgin)
        # Nếu là ảnh màu thì chuyển RGB sang BGR để xử lý bằng OpenCV
        # imgin = imgin[:, :, [2, 1, 0]]
        # Chuyển imgin sang ảnh xám có thể ko cần
        # imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
        if chuong_3_item == 'Negative':
            imgout = Negative(imgin)
        elif chuong_3_item == 'Hist Equal':
            imgout = cv2.equalizeHist(imgin)
        else:
            imgout = LocalHist(imgin)
        # Hiển thị ảnh màu BRG
        # imgout_frame.image(imgin, channels="BGR")
        imgout_frame.image(imgout)
