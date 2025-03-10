# model_config.py
import os
import logging
import threading
from ultralytics import YOLO

# **پیکربندی لاگینگ**
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# **تنظیمات پیش‌فرض**
config = {
    "RESULTS_FOLDER": "results",
    "UPLOAD_FOLDER": "uploads",
    "ALLOWED_EXTENSIONS": ["png", "jpg", "jpeg", "mp4"],
    "RESULTS_TEXT_FILE": "results.txt",
    "MODEL_FOLDER": "runs/detect",
    "DEFAULT_MODEL": "Model 7",
    "AVAILABLE_MODELS": {
        "Model 2": "fire_detection_ros2/fire_detection_ros2/runs/detect/train2/weights/best.pt",
        "Model 4": "fire_detection_ros2/fire_detection_ros2/runs/detect/train9/weights/best.pt",
        "Model 7": "fire_detection_ros2/fire_detection_ros2/runs/detect/train12/weights/best.pt"
    },
    "MAX_CONTENT_LENGTH": 50 * 1024 * 1024  # 50MB - Increased file size limit
}

# Define the order of models for detection (7, 4, 2)
detection_order = [config['DEFAULT_MODEL'], "Model 4", "Model 2"]

# **بارگیری مدل‌های YOLO در startup**
yolo_models = {}
model_lock = threading.Lock()

def load_yolo_models():
    """ بارگیری مدل‌های YOLO و ذخیره در دیکشنری yolo_models. """
    global yolo_models, model_lock
    for model_name, model_path in config['AVAILABLE_MODELS'].items():
        full_model_path = os.path.join(os.getcwd(), model_path)
        if not os.path.exists(full_model_path):
            logging.error(f"Model file not found at path: {full_model_path} for {model_name}")
            continue
        try:
            with model_lock:
                yolo_models[model_name] = YOLO(full_model_path)
            logging.info(f"Model '{model_name}' loaded successfully from {full_model_path}")
        except Exception as e:
            logging.error(f"Error loading model '{model_name}' from {full_model_path}: {e}")

def get_config():
    """ تابع دسترسی به config. """
    return config

def get_detection_order():
    """ تابع دسترسی به detection_order. """
    return detection_order

def get_yolo_models():
    """ تابع دسترسی به yolo_models. """
    return yolo_models

def get_model_lock():
    """ تابع دسترسی به model_lock. """
    return model_lock