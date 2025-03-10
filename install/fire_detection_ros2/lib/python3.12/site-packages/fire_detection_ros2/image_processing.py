# image_processing.py
import cv2
import numpy as np
import logging
from ultralytics import YOLO  # Import YOLO here as it's used in process_image

# **پیکربندی لاگینگ برای این فایل (اختیاری، اگر می‌خواهید لاگ‌های جداگانه داشته باشید)**
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (numpy.ndarray): A numpy array of shape (4,) representing the first bounding box [x1, y1, x2, y2].
        box2 (numpy.ndarray): A numpy array of shape (4,) representing the second bounding box [x1, y1, x2, y2].

    Returns:
        float: The IoU of the two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the (x, y) coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def process_fire_detection(image, detection_order, yolo_models, model_lock):
    """
    Processes the image for fire detection using YOLO models in a cascade.

    Args:
        image (numpy.ndarray): The input image.
        detection_order (list): List of model names to use for detection in order.
        yolo_models (dict): Dictionary of loaded YOLO models.
        model_lock (threading.Lock): Lock for thread-safe access to models.

    Returns:
        tuple: A tuple containing the processed image and detection results.
               Returns (None, None) if no fire is detected or an error occurs.
    """
    original_height, original_width = image.shape[:2]

    # **تغییر سایز تصویر به 640x640**
    resized_image = cv2.resize(image, (640, 640))
    height, width, channels = resized_image.shape
    slice_height = 640 // 2  # **تغییر: 3 قسمت افقی**

    all_results = []  # List to store results from all parts

    # **پردازش هر سه قسمت تصویر (بالا، وسط، پایین)**
    for i in range(2):  # **تغییر: 3 قسمت**
        start_row = i * slice_height
        end_row = (i + 1) * slice_height
        start_col = 0
        end_col = width  # کل عرض

        cropped_part = resized_image[start_row:end_row, start_col:end_col]

        # **ایجاد یک عکس 640x640 جدید با پس‌زمینه خالی (سیاه) برای هر قسمت**
        processed_input_image = np.zeros((640, 640, channels), dtype=np.uint8)
        processed_input_image[0:slice_height, 0:width] = cropped_part  # قرار دادن قسمت در بالا

        # **پردازش YOLO روی قسمت فعلی**
        results = None
        detected_model = None
        for model_name in detection_order:
            if model_name not in yolo_models:
                logging.warning(f"Model '{model_name}' not loaded. Skipping.")
                continue

            with model_lock:
                results = yolo_models[model_name](processed_input_image)

            if results and results[0].boxes.shape[0] > 0:
                detected_model = model_name
                break

        if detected_model and results and results[0].boxes.shape[0] > 0:
            # **دریافت باکس‌ها و تنظیم مختصات برای قسمت فعلی**
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                # **تغییر مختصات باکس‌ها به مکان اصلی در تصویر 640x640**
                adjusted_x1 = x1 + start_col
                adjusted_y1 = y1 + start_row
                adjusted_x2 = x2 + start_col
                adjusted_y2 = y2 + start_row
                all_results.append({
                    'box': [adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'model_name': detected_model
                })

    # **فیلتر کردن باکس‌های هم‌پوشان از تمام قسمت‌ها**
    filtered_results_final = []
    if all_results:
        boxes_all = np.array([res['box'] for res in all_results])
        confidences_all = np.array([res['confidence'] for res in all_results])
        class_ids_all = np.array([res['class_id'] for res in all_results])
        model_names_all = [res['model_name'] for res in all_results]

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
            filtered_results_final.append({
                'box': boxes_all[i],
                'confidence': confidences_all[i],
                'class_id': class_ids_all[i],
                'model_name': model_names_all[i]
            })

    # **رسم باکس‌ها روی تصویر اصلی 640x640 (resized_image)**
    processed_image_resized = resized_image.copy()  # تغییر نام متغیر برای وضوح بیشتر
    if filtered_results_final:
        for res in filtered_results_final:
            box = res['box']
            confidence = res['confidence']
            class_id = int(res['class_id'])
            model_name = res['model_name']
            if model_name == "Model 4":
                label = f"fire {confidence:.2f}"  # لیبل رو به "fire" تغییر بده
            else:
                label = f"{yolo_models[model_name].names[class_id]} {confidence:.2f}"  # لیبل پیش‌فرض
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # رسم روی resized_image
            cv2.putText(processed_image_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # **تغییر سایز تصویر پردازش شده به ابعاد اصلی**
    processed_image_original_size = cv2.resize(processed_image_resized, (original_width, original_height))

    return processed_image_original_size, filtered_results_final