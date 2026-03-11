import os
import json
import numpy as np
from datetime import datetime


def ensure_dir(path: str):
    """Klasör yoksa oluşturur."""
    os.makedirs(path, exist_ok=True)


def save_landmarks(landmarks: np.ndarray, path: str):
    """Landmark dizisini .npy formatında kaydeder."""
    ensure_dir(os.path.dirname(path))
    np.save(path, landmarks)


def load_landmarks(path: str) -> np.ndarray:
    """Kaydedilmiş landmark dizisini yükler."""
    return np.load(path)


def save_json(data: dict, path: str):
    """Dict'i JSON olarak kaydeder."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    """JSON dosyasını yükler."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Zaman damgası döndürür."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Referans tabanlı normalizasyon.
    Tüm koordinatları 0-1 arasına ölçekler.
    """
    if np.all(landmarks == 0):
        return landmarks
    min_val = landmarks.min()
    max_val = landmarks.max()
    if max_val - min_val < 1e-6:
        return landmarks
    return (landmarks - min_val) / (max_val - min_val)


class Logger:
    """Basit loglama sınıfı."""

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        ensure_dir(log_dir)
        timestamp = get_timestamp()
        self.log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")

    def info(self, message: str):
        self.log(message, "INFO")

    def error(self, message: str):
        self.log(message, "ERROR")

    def warning(self, message: str):
        self.log(message, "WARNING")