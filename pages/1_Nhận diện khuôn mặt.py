from pathlib import Path
import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os
from datetime import datetime

st.title('Nhận dạng khuôn mặt')
FRAME_WINDOW = st.image([])

if 'capturing' not in st.session_state:
    st.session_state.capturing = True
    st.session_state.cap = cv.VideoCapture(0)
    st.session_state.button_text = "Stop"

if 'frame_stop_img' not in st.session_state:
    frame_stop_img_data = cv.imread('../stop.jpg')
    if frame_stop_img_data is None:
        frame_stop_img_data = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(frame_stop_img_data, "CAMERA STOPPED", (80, 240), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
                   cv.LINE_AA)
    st.session_state.frame_stop_img = frame_stop_img_data

if st.button(st.session_state.button_text):
    if st.session_state.capturing:
        st.session_state.capturing = False
        if st.session_state.cap.isOpened():
            st.session_state.cap.release()
        st.session_state.button_text = "Start"
    else:
        st.session_state.capturing = True
        st.session_state.cap = cv.VideoCapture(0)
        st.session_state.button_text = "Stop"
    st.rerun()

if not st.session_state.capturing:
    FRAME_WINDOW.image(st.session_state.frame_stop_img, channels='BGR')
else:
    try:
        svc = joblib.load('models/face_recognition/svc.pkl')
        mydict = ['myph', 'ngocanh', 'tuyen', 'uyen', 'yang']
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.error("Vui lòng đảm bảo file 'models/face_recognition/svc.pkl' tồn tại và đúng định dạng.")
        st.stop()


    def save_face_image(frame, face, label):
        try:
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[0], coords[1], coords[2], coords[3]
            pad = 20
            y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                save_dir = "recognized_faces"
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{save_dir}/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv.imwrite(filename, face_crop)
        except Exception as e:
            if 'sidebar' in st.elements:  # Check if sidebar exists before trying to use it
                st.sidebar.error(f"Lỗi khi lưu ảnh: {e}")
            else:
                print(f"Lỗi khi lưu ảnh (không có sidebar): {e}")


    def visualize(input_img, faces, fps, thickness=2, names=None):
        img_height, img_width = input_img.shape[:2]

        if faces[1] is not None:
            if names is not None and not isinstance(names, (list, tuple)):
                names_list = [names]
            else:
                names_list = names if names is not None else []

            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]

                cv.rectangle(input_img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), thickness)

                for i in range(5):
                    cv.circle(input_img, (coords[4 + i * 2], coords[5 + i * 2]), 2, (0, 255, 255), thickness)

                if names_list and idx < len(names_list):
                    label = names_list[idx]
                    font = cv.FONT_HERSHEY_SIMPLEX
                    base_font_scale = 2.5
                    thickness_text = 4
                    padding = 15
                    text_color = (255, 102, 255) if label != "Unknow" else (0, 0, 255)
                    bg_color = (0, 0, 0)
                    max_text_width = min(w + 40, img_width - 2 * padding)
                    font_scale = base_font_scale
                    while True:
                        (text_w, text_h), base = cv.getTextSize(label, font, font_scale, thickness_text)
                        if text_w <= max_text_width or font_scale <= 0.2:
                            break
                        font_scale -= 0.1

                    box_y2 = y1 - padding // 2
                    box_y1 = box_y2 - text_h - base - padding
                    if box_y1 < 0:
                        box_y1 = y1 + h + padding
                        box_y2 = box_y1 + text_h + base + padding

                    text_x_center = x1 + w // 2
                    box_x1 = text_x_center - text_w // 2 - padding
                    box_x2 = text_x_center + text_w // 2 + padding
                    box_x1 = max(0, box_x1)
                    box_x2 = min(img_width, box_x2)
                    text_x_in_box = box_x1 + padding
                    text_y_in_box = box_y2 - base - padding // 2

                    if box_y2 > box_y1 and box_x2 > box_x1:
                        cv.rectangle(input_img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
                        cv.putText(input_img, label, (text_x_in_box, text_y_in_box), font, font_scale, text_color,
                                   thickness_text, cv.LINE_AA)

        cv.putText(input_img, f'FPS: {fps:.2f}', (15, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv.LINE_AA)


    try:
        detector = cv.FaceDetectorYN.create(
            'models/face_recognition/face_detection_yunet_2023mar.onnx', "", (320, 320), 0.9, 0.3, 5000)
        recognizer = cv.FaceRecognizerSF.create(
            'models/face_recognition/face_recognition_sface_2021dec.onnx', "")
    except Exception as e:
        st.error(f"Lỗi tạo detector hoặc recognizer: {e}")
        st.error("Vui lòng đảm bảo các file model .onnx tồn tại trong 'models/face_recognition/'.")
        st.stop()

    try:
        _ = svc.predict_proba(np.zeros((1, 128)))
        support_proba = True
    except AttributeError:
        support_proba = False
    except Exception as e:
        st.warning(f"Không thể kiểm tra predict_proba, giả định là không hỗ trợ: {e}")
        support_proba = False

    tm = cv.TickMeter()

    current_cap = st.session_state.cap
    if current_cap.isOpened():
        w = int(current_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(current_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            detector.setInputSize([w, h])
        else:
            st.warning("Không lấy được kích thước frame, đặt mặc định 640x480 cho detector.")
            detector.setInputSize([640, 480])
    else:
        if st.session_state.capturing:
            st.error("Không thể mở camera. Vui lòng kiểm tra lại camera của bạn.")
            st.session_state.capturing = False
            st.session_state.button_text = "Start"
            st.rerun()

    last_recognized_name = None
    name_recognized_time = None
    SAVE_IMAGE_COOLDOWN = 5

    while st.session_state.capturing and current_cap.isOpened():
        has_frame, frame = current_cap.read()
        if not has_frame or frame is None:
            st.warning("Không nhận được frame từ camera. Đang thử lại...")
            cv.waitKey(100)
            continue

        frame_copy = frame.copy()
        tm.start()
        faces = detector.detect(frame_copy)
        tm.stop()
        detected_names = []

        if faces[1] is not None and len(faces[1]) > 0:
            for face_info in faces[1]:
                current_result = "Unknow"
                try:
                    aligned_face = recognizer.alignCrop(frame_copy, face_info)
                    if aligned_face is not None and aligned_face.size > 0:
                        feature = recognizer.feature(aligned_face)
                        pred_index = svc.predict(feature)[0]

                        if 0 <= pred_index < len(mydict):
                            confidence_threshold_proba = 0.80
                            confidence_threshold_decision = 0.55

                            if support_proba:
                                probabilities = svc.predict_proba(feature)[0]
                                prob = np.max(probabilities)
                                if prob >= confidence_threshold_proba:
                                    current_result = mydict[pred_index]
                            else:
                                decision_values = svc.decision_function(feature)[0]
                                if hasattr(svc, 'classes_'):
                                    class_index_in_svc = np.where(svc.classes_ == pred_index)[0]
                                    if class_index_in_svc.size > 0:
                                        if isinstance(decision_values,
                                                      np.ndarray) and decision_values.ndim == 1 and decision_values.size == len(
                                                svc.classes_):
                                            score = decision_values[class_index_in_svc[0]]
                                        elif isinstance(decision_values, np.ndarray) and decision_values.ndim == 0:
                                            score = decision_values if pred_index == 1 else -decision_values
                                        else:
                                            score = 0
                                        if score >= confidence_threshold_decision:
                                            current_result = mydict[pred_index]
                                else:
                                    current_result = "Unknow"
                        else:
                            current_result = "Unknow"

                        if current_result != "Unknow":
                            now = datetime.now()
                            save_this_face = False
                            if current_result != last_recognized_name:
                                save_this_face = True
                            elif name_recognized_time and (
                                    now - name_recognized_time).total_seconds() > SAVE_IMAGE_COOLDOWN:
                                save_this_face = True

                            if save_this_face:
                                save_face_image(frame, face_info, current_result)
                                last_recognized_name = current_result
                                name_recognized_time = now
                    else:
                        current_result = "Unknow"
                except Exception as e:
                    current_result = "Unknow"
                detected_names.append(current_result)
        else:
            last_recognized_name = None
            name_recognized_time = None

        visualize(frame_copy, faces, tm.getFPS(), names=detected_names if detected_names else None)
        FRAME_WINDOW.image(frame_copy, channels='BGR')

    if not st.session_state.capturing and current_cap.isOpened():
        current_cap.release()