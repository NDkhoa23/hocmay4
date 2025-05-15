import cv2
from ultralytics import YOLO
import os

# Đường dẫn đến ảnh
image_path = r"dataset\images\test\video1_frame_00015.jpg"  # Thay đổi theo đường dẫn ảnh của bạn

# Đọc ảnh
image = cv2.imread(image_path)

# Kiểm tra nếu ảnh được đọc thành công
if image is None:
    print("Không thể đọc ảnh")
    exit()


# Load mô hình YOLOv8 đã huấn luyện
model = YOLO("runs/detect/train5/weights/best.pt")  # Đường dẫn tới mô hình của bạn

# Phát hiện đối tượng bằng YOLO
results = model(image)[0]

# Vẽ bounding box và nhãn cho các đối tượng
for box in results.boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    conf = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Vẽ bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    short_label = label.split("_")[-1].rstrip("s")
    label_text = f"{short_label} {conf:.2f}"
    # Hiển thị nhãn và độ chính xác
    
    cv2.putText(image, label_text, (x1+5, y1 -5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Hiển thị ảnh đã xử lý
cv2.imshow("Processed Image", image)

# Lưu ảnh đã xử lý

# Chờ phím 'q' để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
