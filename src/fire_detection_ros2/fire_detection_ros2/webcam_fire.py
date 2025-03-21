import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import logging
import time
import threading
from fire_detection_ros2.model_config import load_yolo_models, get_detection_order, get_yolo_models, get_model_lock
from fire_detection_ros2.image_processing import iou

# پیکربندی لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FireDetectorNode(Node):
    def __init__(self):
        super().__init__('fire_detector_node')
        self.get_logger().info("FireDetectorNode initialized and subscribing to /image_raw")
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.image_subscription
        self.processed_image_publisher_ = self.create_publisher(
            Image,
            '/fire_detection/processed_image',  # تاپیک با namespace
            10)

        # بارگیری پیکربندی و مدل‌ها
        load_yolo_models()
        self.detection_order = get_detection_order()
        self.yolo_models = get_yolo_models()
        self.model_lock = get_model_lock()

        # متغیرهای ردیابی و ترکیب فریم
        self.previous_frame = None
        self.tracked_boxes = []

        # تنظیم پنجره نمایش
        self.window_name = 'Fire Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def process_frame_section(self, frame, start_row, end_row):
        """پردازش بخشی از فریم برای تشخیص اشیا."""
        height, width, channels = frame.shape
        cropped_part = frame[start_row:end_row, 0:width]
        processed_input_image = np.zeros((640, 640, channels), dtype=np.uint8)
        processed_input_image[0:(end_row - start_row), 0:width] = cropped_part

        results = None
        detected_model = None
        for model_name in self.detection_order:
            if model_name not in self.yolo_models:
                logging.warning(f"Model '{model_name}' not loaded. Skipping.")
                continue
            with self.model_lock:
                results = self.yolo_models[model_name](processed_input_image)
            if results and results[0].boxes.shape[0] > 0:
                detected_model = model_name
                break

        frame_results = []
        if detected_model and results and results[0].boxes.shape[0] > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence < 0.3:  # فیلتر اعتماد به نفس پایین
                    continue
                x1, y1, x2, y2 = map(int, box)
                adjusted_x1 = x1
                adjusted_y1 = y1 + start_row
                adjusted_x2 = x2
                adjusted_y2 = y2 + start_row
                frame_results.append({
                    'box': [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'model_name': detected_model,
                    'age': 0  # برای ردیابی
                })

        return frame_results

    def iou(self, box1, box2):
        """محاسبه IOU بین دو باکس."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        iou_value = inter_area / float(box1_area + box2_area - inter_area)
        return iou_value

    def distance_between_boxes(self, box1, box2):
        """محاسبه فاصله مرکزی بین دو باکس."""
        center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def merge_and_track_boxes(self, new_boxes, max_age=5, distance_threshold=100, iou_threshold=0.3):
        """ادغام و ردیابی باکس‌ها با فیلتر همپوشانی."""
        if not new_boxes:
            updated_tracked_boxes = []
            for box in self.tracked_boxes:
                box['age'] += 1
                if box['age'] <= max_age:
                    updated_tracked_boxes.append(box)
            return updated_tracked_boxes

        filtered_new_boxes = []
        for i, new_box in enumerate(new_boxes):
            keep = True
            for j, other_box in enumerate(filtered_new_boxes):
                if self.iou(new_box['box'], other_box['box']) > iou_threshold:
                    if new_box['confidence'] > other_box['confidence']:
                        filtered_new_boxes[j] = new_box
                    keep = False
                    break
            if keep:
                filtered_new_boxes.append(new_box)

        updated_tracked_boxes = []
        used_new_boxes = set()

        for tracked_box in self.tracked_boxes:
            tracked_box['age'] += 1
            min_distance = float('inf')
            best_match = None

            for i, new_box in enumerate(filtered_new_boxes):
                if i in used_new_boxes:
                    continue
                dist = self.distance_between_boxes(tracked_box['box'], new_box['box'])
                if dist < min_distance and dist < distance_threshold:
                    min_distance = dist
                    best_match = i

            if best_match is not None:
                new_box = filtered_new_boxes[best_match]
                tracked_box['box'] = new_box['box']
                tracked_box['confidence'] = new_box['confidence']
                tracked_box['class_id'] = new_box['class_id']
                tracked_box['model_name'] = new_box['model_name']
                tracked_box['age'] = 0
                used_new_boxes.add(best_match)
                updated_tracked_boxes.append(tracked_box)
            elif tracked_box['age'] <= max_age:
                updated_tracked_boxes.append(tracked_box)

        for i, new_box in enumerate(filtered_new_boxes):
            if i not in used_new_boxes:
                updated_tracked_boxes.append(new_box)

        return updated_tracked_boxes

    def overlay_two_frames(self, frame1, frame2, alpha=0.5):
        """ترکیب دو فریم با شفافیت 50%."""
        combined_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0.0)
        return combined_frame

    def draw_boxes(self, frame):
        """رسم باکس‌ها روی فریم."""
        for res in self.tracked_boxes:
            box = res['box']
            confidence = res['confidence']
            class_id = int(res['class_id'])
            model_name = res['model_name']
            label = f"fire {confidence:.2f}" if model_name == "Model 4" else f"{self.yolo_models[model_name].names[class_id]} {confidence:.2f}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def image_callback(self, msg):
        self.get_logger().info("Image received on /image_raw topic!")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f"Image converted to cv2 successfully. Shape: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f'Could not convert image message to cv2: {e}')
            return

        # تغییر اندازه فریم
        resized_frame = cv2.resize(cv_image, (640, 640))

        # ترکیب با فریم قبلی
        if self.previous_frame is not None:
            combined_frame = self.overlay_two_frames(self.previous_frame, resized_frame, alpha=0.5)
        else:
            combined_frame = resized_frame

        # پردازش فریم ترکیبی
        height, width, channels = combined_frame.shape
        upper_70_height = int(height * 0.7)
        lower_70_height = int(height * 0.7)
        upper_start = 0
        upper_end = upper_70_height
        lower_start = height - lower_70_height
        lower_end = height

        # پردازش موازی دو بخش
        frame_results = []
        upper_thread = threading.Thread(target=lambda: frame_results.extend(self.process_frame_section(combined_frame, upper_start, upper_end)))
        lower_thread = threading.Thread(target=lambda: frame_results.extend(self.process_frame_section(combined_frame, lower_start, lower_end)))

        upper_thread.start()
        lower_thread.start()
        upper_thread.join()
        lower_thread.join()

        # به‌روزرسانی باکس‌های ردیابی‌شده
        self.tracked_boxes = self.merge_and_track_boxes(frame_results, max_age=5, distance_threshold=100, iou_threshold=0.3)

        # رسم باکس‌ها روی فریم
        processed_frame = self.draw_boxes(resized_frame)

        # انتشار تصویر پردازش‌شده
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            self.processed_image_publisher_.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f'Could not convert cv2 image to image message: {e}')

        # نمایش تصویر
        cv2.imshow(self.window_name, processed_frame)
        cv2.waitKey(1)

        # ذخیره فریم فعلی به‌عنوان فریم قبلی
        self.previous_frame = resized_frame.copy()

    def destroy_node(self):
        """تمیز کردن منابع هنگام خاموش شدن نود."""
        super().destroy_node()
        cv2.destroyAllWindows()
        self.get_logger().info("FireDetectorNode destroyed and windows closed.")

def main(args=None):
    rclpy.init(args=args)
    fire_detector_node = FireDetectorNode()
    try:
        rclpy.spin(fire_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        fire_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()