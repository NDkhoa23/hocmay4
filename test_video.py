import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load mô hình YOLOv8 đã huấn luyện
model = YOLO("runs/detect/train5/weights/best.pt")

# Đọc video
video_path = "video4.mp4"  # Đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu video được mở thành công
if not cap.isOpened():
    print("Không thể mở video")
    exit()

# Lấy thông tin về video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Tạo tên file đầu ra từ tên video gốc (lấy tên file mà không có phần mở rộng)
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = f"output_{video_name}.mp4"  # Định dạng tên file: output_video_video1.mp4

# Kiểm tra nếu thư mục đầu ra tồn tại, nếu không thì tạo thư mục
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo VideoWriter để lưu video đã xử lý
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("Không thể tạo video output")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if video_path == "video1.mp4":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Phát hiện đối tượng bằng YOLO
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Rút gọn nhãn: lấy phần sau cùng sau dấu "_" và bỏ "s" nếu có
        short_label = label.split("_")[-1].rstrip("s")
        label_text = f"{short_label} {conf:.2f}"

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Hiển thị nhãn rút gọn + độ chính xác
        cv2.putText(frame, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Lưu video đã xử lý
    out.write(frame)

    # Hiển thị video đã xử lý
    cv2.imshow("Processed Video", frame)

    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video đã được lưu tại: {output_path}")
