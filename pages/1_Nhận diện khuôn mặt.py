from pathlib import Path
import streamlit as st
import numpy as np
import cv2 as cv
import joblib

st.subheader('Nhận dạng khuôn mặt')
FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

press = st.button('Stop')
if press:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False
        # Re-initialize capture if stopping was undone
        cap = cv.VideoCapture(0)  # Thêm dòng này để khởi tạo lại camera khi bấm Stop lần nữa

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('../stop.jpg')
    # Kiểm tra nếu stop.jpg không load được
    if frame_stop is None:
        # Tạo ảnh đen thay thế nếu không có stop.jpg
        frame_stop = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(frame_stop, "CAMERA STOPPED", (100, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print('Warning: stop.jpg not found, using fallback image.')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg (hoặc ảnh thay thế)')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
# --- Chỉ thực thi phần còn lại nếu camera không bị stop ---
elif not st.session_state.stop:

    # --- Tải model và danh sách tên ---
    try:
        svc = joblib.load('models/face_recognition/svc.pkl')
        mydict = ['myph', 'nhan', 'tuyen', 'uyen', 'yang']  # Đảm bảo danh sách này khớp với model svc.pkl
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file svc.pkl. Vui lòng đảm bảo file model tồn tại.")
        st.stop()  # Dừng ứng dụng nếu không có model
    except Exception as e:
        st.error(f"Lỗi khi tải model: {e}")
        st.stop()


    # ================== HÀM VISUALIZE ĐÃ CHỈNH SỬA THEO STYLE CODE 2 ==================
    def visualize(input, faces, fps, recognized_name=None, thickness=2):
        img_height, img_width = input.shape[:2]  # Lấy kích thước ảnh để clipping

        if faces[1] is not None:
            # Chỉ xử lý khuôn mặt đầu tiên (do code 1 chỉ recognize 1 khuôn mặt)
            if len(faces[1]) > 0:
                idx = 0  # Chỉ số của khuôn mặt đầu tiên
                face = faces[1][idx]

                # Vẽ bounding box và landmarks (giữ nguyên từ code 1 gốc)
                coords = face[:-1].astype(np.int32)
                x1, y1, w, h = coords[0], coords[1], coords[2], coords[3]
                cv.rectangle(input, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

                # --- Phần hiển thị tên đã copy và chỉnh sửa từ code 2 ---
                if recognized_name:  # Chỉ vẽ nếu có tên được truyền vào
                    label = recognized_name

                    # Cài đặt font chữ giống code 2
                    fontFace = cv.FONT_HERSHEY_SIMPLEX
                    fontScale = 1.7  # Kích thước chữ lớn hơn
                    textThickness = 3  # Độ dày chữ (tạo hiệu ứng đậm)
                    padding = 10  # Khoảng đệm giữa chữ và viền khung nền

                    # Xác định màu chữ (Giống code 2)
                    if label == "Unknow":
                        textColor = (0, 0, 255)  # Màu đỏ (BGR) cho Unknow
                    else:
                        textColor = (0, 255, 0)  # Màu xanh lá (BGR) cho tên đã biết

                    # Màu nền giống code 2
                    bgColor = (255, 0, 255)  # Màu tím (BGR)

                    # Tính toán kích thước của text
                    (text_width, text_height), baseline = cv.getTextSize(label, fontFace, fontScale, textThickness)
                    baseline += textThickness  # Điều chỉnh baseline

                    # Tính toán vị trí cho khung nền và text (Đặt phía trên bounding box)
                    text_x = x1 + (w - text_width) // 2
                    box_y2 = y1 - padding
                    box_y1 = box_y2 - text_height - baseline - (2 * padding)
                    text_y = box_y2 - baseline - padding
                    box_x1 = text_x - padding
                    box_x2 = text_x + text_width + padding

                    # --- Clipping: Đảm bảo không vẽ ra ngoài ảnh ---
                    box_x1 = max(0, box_x1)
                    box_y1 = max(0, box_y1)
                    box_x2 = min(img_width, box_x2)
                    box_y2 = min(img_height, box_y2)
                    text_x = box_x1 + padding  # Điều chỉnh lại text_x sau khi clip box_x1
                    # Điều chỉnh text_y nếu box_y1 bị clip
                    text_y = max(box_y1 + padding + baseline + text_height, text_y)

                    # --- Vẽ khung nền và text (Giống code 2) ---
                    if box_y1 < box_y2 and box_x1 < box_x2:  # Chỉ vẽ nếu khung hợp lệ
                        cv.rectangle(input, (box_x1, box_y1), (box_x2, box_y2), bgColor, -1)  # -1 để tô đầy
                        cv.putText(input, label, (text_x, text_y), fontFace, fontScale, textColor, textThickness,
                                   cv.LINE_AA)

            # Vẽ các khuôn mặt khác nếu có (chỉ vẽ bounding box, không vẽ tên)
            for idx, face in enumerate(faces[1]):
                if idx > 0:  # Bỏ qua khuôn mặt đầu tiên đã xử lý ở trên
                    coords = face[:-1].astype(np.int32)
                    cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]),
                                 (0, 255, 0), thickness)
                    # Có thể vẽ thêm landmarks cho các khuôn mặt khác nếu muốn
                    # cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                    # ...

        # Hiển thị FPS (giữ nguyên từ code 1 gốc)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # ==============================================================================

    # --- Phần khởi tạo detector và recognizer ---
    try:
        detector = cv.FaceDetectorYN.create(
            'models/face_recognition/face_detection_yunet_2023mar.onnx',
            "",
            (320, 320),
            0.9,  # score_threshold
            0.3,  # nms_threshold
            5000  # top_k
        )
        recognizer = cv.FaceRecognizerSF.create(
            'models/face_recognition/face_recognition_sface_2021dec.onnx', ""
        )
    except cv.error as e:
        st.error(f"Lỗi OpenCV khi khởi tạo model nhận dạng: {e}. Kiểm tra đường dẫn file .onnx.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi không xác định khi khởi tạo model: {e}")
        st.stop()

    tm = cv.TickMeter()

    # Lấy kích thước frame và đặt cho detector
    if cap.isOpened():
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        if frameWidth > 0 and frameHeight > 0:
            detector.setInputSize([frameWidth, frameHeight])
        else:
            st.warning("Không lấy được kích thước frame từ camera, sử dụng kích thước mặc định 320x320 cho detector.")
            detector.setInputSize([320, 320])  # Fallback
    else:
        st.error("Không thể mở camera.")
        st.stop()

    # --- Vòng lặp chính xử lý frame ---
    while not st.session_state.stop and cap.isOpened():
        hasFrame, frame = cap.read()
        if not hasFrame:
            st.warning('Không nhận được frame từ camera hoặc camera đã đóng.')
            # Có thể thử khởi tạo lại camera ở đây nếu muốn tự động kết nối lại
            # cap.release()
            # cap = cv.VideoCapture(0)
            # if not cap.isOpened():
            #     st.error("Không thể kết nối lại camera.")
            #     st.session_state.stop = True # Dừng nếu không kết nối lại được
            #     FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
            # continue # Bỏ qua vòng lặp này và thử lại
            break  # Hoặc thoát vòng lặp

        # Đảm bảo frame hợp lệ
        if frame is None or frame.size == 0:
            continue

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        result = None  # Khởi tạo tên nhận dạng là None cho mỗi frame
        if faces[1] is not None:
            # Chỉ nhận dạng khuôn mặt đầu tiên tìm thấy (giống logic gốc của code 1)
            if len(faces[1]) > 0:
                try:
                    face_align = recognizer.alignCrop(frame, faces[1][0])
                    # Kiểm tra kết quả alignCrop
                    if face_align is not None and face_align.size > 0:
                        face_feature = recognizer.feature(face_align)
                        test_predict = svc.predict(face_feature)
                        predicted_index = test_predict[0]

                        # Kiểm tra index có hợp lệ không
                        if 0 <= predicted_index < len(mydict):
                            result = mydict[predicted_index]
                        else:
                            result = "Unknow"  # Nếu index không nằm trong danh sách
                    else:
                        result = "Unknow"  # Nếu alignCrop thất bại
                except cv.error as e:
                    print(f"OpenCV error during recognition: {e}")
                    result = "Unknow"  # Gán Unknow nếu có lỗi OpenCV
                except Exception as e:
                    print(f"Error during recognition: {e}")
                    result = "Unknow"  # Gán Unknow nếu có lỗi khác

        # --- Vẽ kết quả lên frame bằng hàm visualize đã sửa ---
        # Truyền tên nhận dạng (result) vào hàm visualize
        visualize(frame, faces, tm.getFPS(), recognized_name=result)

        # --- XÓA dòng cv.putText cũ đi ---
        # # cv.putText(frame, result, (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Dòng này đã bị thay thế bởi logic trong visualize

        # Hiển thị frame lên Streamlit
        FRAME_WINDOW.image(frame, channels='BGR')

    # Giải phóng camera khi vòng lặp kết thúc (do stop=True hoặc lỗi)
    if cap.isOpened():
        cap.release()
    print("Camera released.")
# cv.destroyAllWindows() # Không cần thiết trong Streamlit

