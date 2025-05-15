import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/train5/weights/best.pt")
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Đặt tesseract nếu không có trong PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nếu video bị quay ngang (xoay 90 độ)
    # Bạn có thể xoay lại frame theo hướng đúng trước khi xử lý
    if video_path == "video1.mp4":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Hoặc dùng ROTATE_90_COUNTERCLOCKWISE nếu video quay ngược chiều kim đồng hồ

    # Phát hiện bằng YOLO
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Nếu là vùng countdown thì OCR
        if label == "countdown_number":
            roi = frame[y1:y2, x1:x2]

            # Chuyển sang không gian màu HSV để dễ dàng lọc màu
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Lọc màu xanh (dải màu trong HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([150, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)

            # Lọc màu đỏ (dải màu trong HSV)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Kết hợp các mask màu xanh và đỏ
            mask = cv2.bitwise_or(green_mask, red_mask)

            # Áp dụng mask để chỉ giữ lại các phần có màu xanh hoặc đỏ
            result = cv2.bitwise_and(roi, roi, mask=mask)

            # Chuyển về ảnh đen trắng cho Tesseract dễ nhận diện hơn
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 117, 255, cv2.THRESH_BINARY)

            # Sử dụng pytesseract để nhận diện chữ
            text = pytesseract.image_to_string(thresh, config='--psm 7 digits')
            text = text.replace("O", "0")
            print("Countdown detected:", text.strip())

            # Hiển thị kết quả lên video
            cv2.putText(frame, f"Countdown: {text.strip()}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Vẽ khung các object


    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
