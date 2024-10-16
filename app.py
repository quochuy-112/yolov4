from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(r"D:\yolov4\darknet")
import darknet
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)


# Đường dẫn đến file cấu hình, trọng số và file dữ liệu (chứa thông tin class)
config_path = r"C:\xampp\htdocs\yolov4\yolov4-custom.cfg"
weights_path = r"C:\xampp\htdocs\yolov4\yolov4-custom_last.weights"
data_path = r"C:\xampp\htdocs\yolov4\obj.data"

# Load YOLOv4 từ Darknet
network, class_names, class_colors = darknet.load_network(
    config_path, data_path, weights_path
)

def convert_cv2_to_darknet(image):
    darknet_image = darknet.make_image(image.shape[1], image.shape[0], 3)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(darknet_image, img_rgb.tobytes())
    return darknet_image

def draw_bounding_boxes(image, detections, class_colors, thickness=5):
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        left = int(x - w / 2)
        top = int(y - h / 2)
        right = int(x + w / 2)
        bottom = int(y + h / 2)
        color = class_colors[label]
        confidence = float(confidence)
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
        cv2.putText(image, f"{label} [{confidence:.2f}]", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def detect_objects(image):
    darknet_image = convert_cv2_to_darknet(image)
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
    darknet.free_image(darknet_image)
    image_with_boxes = draw_bounding_boxes(image, detections, class_colors, thickness=3)

    results = []
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        results.append({
            "label": label,
            "confidence": float(confidence),
            "bbox": [x, y, w, h]
        })

    return results, image_with_boxes

def save_image(image, filename):
    # Tạo thư mục output nếu chưa có
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo đường dẫn lưu ảnh
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    filename = file.filename
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    results, processed_img = detect_objects(img)

    saved_image = processed_img.copy()

    if not results:
        results = [{"label": None, "confidence": None, "bbox": None}]
        processed_img = img
        saved_image = img

    # Mã hóa ảnh để hiển thị trên web trước khi lưu
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Lưu ảnh đã xử lý
    save_image(saved_image, f"detected_{filename}")

    print("Detected objects:", results)
    print("Base64 image length:", len(img_base64))

    return jsonify({"objects": results, "image": img_base64})

if __name__ == "__main__":
    app.run(debug=True)