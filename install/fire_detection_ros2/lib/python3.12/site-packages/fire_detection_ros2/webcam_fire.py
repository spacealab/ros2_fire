#webcam_fire.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import logging
import time
from fire_detection_ros2.model_config import load_yolo_models, get_detection_order, get_yolo_models, get_model_lock # import from inside package
from fire_detection_ros2.image_processing import iou # import from inside package
# from image_transport import ImageTransport

# **پیکربندی لاگینگ**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FireDetectorNode(Node): # کلاس نود ROS2
    def __init__(self):
        super().__init__('fire_detector_node')
        self.get_logger().info("FireDetectorNode initialized and subscribing to /image_raw")
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image,
            '/image_raw',  # اضافه کردن / برای تاپیک مطلق
            self.image_callback,
            10)
        self.image_subscription  # جلوگیری از استفاده نشده بودن متغیر
        self.processed_image_publisher_ = self.create_publisher( # پابلیشر برای تصویر پردازش شده (اختیاری)
            Image,
            'processed_image',
            10)

        # **بارگیری پیکربندی و مدل‌ها**
        load_yolo_models()
        self.detection_order = get_detection_order()
        self.yolo_models = get_yolo_models()
        self.model_lock = get_model_lock()

        self.window_name = 'Fire Detection' # نام پنجره نمایش
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL) # پنجره با قابلیت تغییر اندازه

    def image_callback(self, msg):
        self.get_logger().info("Image received on /image_raw topic!")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().info(f"Image converted to cv2 successfully. Shape: {cv_image.shape}")
        except Exception as e:
            self.get_logger().error(f'Could not convert image message to cv2: {e}')
            return
        processed_frame = self.process_frame(cv_image)
        cv2.imshow(self.window_name, processed_frame)
        cv2.waitKey(1)

        # انتشار تصویر پردازش شده (اختیاری)
        try:
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8") # تبدیل تصویر OpenCV به پیام ROS2 Image
            self.processed_image_publisher_.publish(processed_msg) # انتشار پیام
        except Exception as e:
            self.get_logger().error(f'Could not convert cv2 image to image message: {e}')

        cv2.imshow(self.window_name, processed_frame) # نمایش تصویر پردازش شده
        cv2.waitKey(1) # برای به‌روزرسانی پنجره نمایش

    def process_frame(self, frame):
        """
        پردازش یک فریم ویدیو برای تشخیص آتش (همان تابع قبلی با کمی تغییرات).
        """
        resized_frame = cv2.resize(frame, (640, 640))
        height, width, channels = resized_frame.shape
        slice_height = 640 // 2

        frame_results = []

        for i in range(2): # پردازش دو قسمت تصویر
            start_row = i * slice_height
            end_row = (i + 1) * slice_height
            start_col = 0
            end_col = width
            cropped_part = resized_frame[start_row:end_row, start_col:end_col]

            processed_input_image = np.zeros((640, 640, channels), dtype=np.uint8)
            processed_input_image[0:slice_height, 0:width] = cropped_part

            results = None
            detected_model = None
            for model_name in self.detection_order: # استفاده از self.detection_order
                if model_name not in self.yolo_models: # استفاده از self.yolo_models
                    logging.warning(f"Model '{model_name}' not loaded. Skipping.")
                    continue
                with self.model_lock: # استفاده از self.model_lock
                    results = self.yolo_models[model_name](processed_input_image) # استفاده از self.yolo_models
                if results and results[0].boxes.shape[0] > 0:
                    detected_model = model_name
                    break

            if detected_model and results and results[0].boxes.shape[0] > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    adjusted_x1 = x1 + start_col
                    adjusted_y1 = y1 + start_row
                    adjusted_x2 = x2 + start_col
                    adjusted_y2 = y2 + start_row
                    frame_results.append({
                        'box': [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'model_name': detected_model
                    })

        # فیلتر کردن باکس‌های هم‌پوشان (همان کد قبلی)
        filtered_results = []
        if frame_results:
            boxes_all = np.array([res['box'] for res in frame_results])
            confidences_all = np.array([res['confidence'] for res in frame_results])
            class_ids_all = np.array([res['class_id'] for res in frame_results])
            model_names_all = [res['model_name'] for res in frame_results]

            keep_indices = list(range(len(boxes_all)))
            for i in range(len(boxes_all)):
                if i not in keep_indices:
                    continue
                for j in range(i + 1, len(boxes_all)):
                    if j not in keep_indices:
                        continue
                    iou_value = iou(boxes_all[i], boxes_all[j])
                    if iou_value > 0:
                        box_i_area = (boxes_all[i][2] - boxes_all[i][0]) * (boxes_all[i][3] - boxes_all[i][1])
                        box_j_area = (boxes_all[j][2] - boxes_all[j][0]) * (boxes_all[j][3] - boxes_all[j][1])
                        if box_i_area >= box_j_area:
                            keep_indices.remove(j)
                        else:
                            keep_indices.remove(i)
                            break

            for i in keep_indices:
                filtered_results.append({
                    'box': boxes_all[i],
                    'confidence': confidences_all[i],
                    'class_id': class_ids_all[i],
                    'model_name': model_names_all[i]
                })

        processed_frame = resized_frame.copy() # رسم روی فریم تغییر سایز داده شده
        if filtered_results:
            for res in filtered_results:
                box = res['box']
                confidence = res['confidence']
                class_id = int(res['class_id'])
                model_name = res['model_name']
                if model_name == "Model 4":
                    label = f"fire {confidence:.2f}"
                else:
                    label = f"{self.yolo_models[model_name].names[class_id]} {confidence:.2f}" # استفاده از self.yolo_models
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return processed_frame

def main(args=None):
    rclpy.init(args=args)
    fire_detector_node = FireDetectorNode() # ایجاد نود
    rclpy.spin(fire_detector_node) # اجرای نود (مانند حلقه while True)

    # بعد از پایان اجرای نود (مثلاً با Ctrl+C)
    fire_detector_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows() # بستن پنجره OpenCV

if __name__ == '__main__':
    main()