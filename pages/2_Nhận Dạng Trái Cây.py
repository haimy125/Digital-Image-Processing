import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

st.write('## Phát hiện trái cây trong ảnh')

model = YOLO('best_train_5_trai_cay.pt')

img_file_buffer = st.sidebar.file_uploader("Upload ảnh trái cây", type=["bmp", "png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
org_frame = col1.empty()
ann_frame = col2.empty()

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    # Chuyển sang cv2 để dùng sau này
    frame = np.array(image)

    # Chuyển từ RGB (PIL) sang BGR (CV2/YOLOv8 input)
    frame = frame[:, :, [2, 1, 0]]  # RGB -> BGR

    # Hiển thị ảnh gốc
    org_frame.image(image)

    # Sử dụng button để chạy dự đoán
    if st.sidebar.button('Predict'):
        names = model.names

        # **Đặt ở đây để đảm bảo frame là liên tục trước khi Annotator xử lý**
        frame = np.ascontiguousarray(frame)

        # Tạo Annotator với frame BGR
        annotator = Annotator(frame)

        # Chạy dự đoán
        results = model.predict(frame, conf=0.6, verbose=False)

        # Lấy kết quả từ result đầu tiên (thường chỉ có 1 ảnh khi dự đoán 1 ảnh)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            confs = results[0].boxes.conf.tolist()

            # Vẽ bounding box và label
            for box, cls, conf in zip(boxes, clss, confs):
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(box, label=label, txt_color=(255, 255, 255), color=(255, 0, 0))

            # Hiển thị ảnh đã vẽ bounding box.
            ann_frame.image(frame, channels="BGR")
        else:
            st.sidebar.write("Không phát hiện thấy trái cây nào.")
            ann_frame.image(frame, channels="BGR")
else:
    st.sidebar.write("Vui lòng upload ảnh để bắt đầu.")