import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from configs.config import Config
from src.video.video_stream_manager import VideoStreamManager
from src.features.feature_extractor import MediaPipeExtractor
from src.models.temporal_classifier import GRUClassifier
from src.translation.translation_engine import TranslationEngine
from src.utils.helpers import Logger


def test_dataset():
    """Dataset'in doğru yüklendiğini test eder."""
    from src.features.dataset_loader import AUTSLDataset

    config = Config()
    logger = Logger()

    logger.info("=== Dataset Test ===")

    # Train dataset'ten bir örnek yükle
    dataset = AUTSLDataset(
        csv_path=config.TRAIN_CSV,
        dataset_root=config.DATASET_ROOT,
        max_seq_length=config.MAX_SEQ_LENGTH,
        cache_data=False
    )

    # İlk örneği al
    sequence, label = dataset[0]
    logger.info(f"Sekans boyutu: {sequence.shape}")
    logger.info(f"Etiket: {label.item()}")
    logger.info(f"Min değer: {sequence.min():.4f}")
    logger.info(f"Max değer: {sequence.max():.4f}")
    logger.info(f"Sıfır olmayan: {(sequence != 0).sum().item()}")
    logger.info("Dataset testi başarılı! ✅")


def test_model():
    """Model mimarisinin doğru çalıştığını test eder."""
    import torch
    config = Config()
    logger = Logger()

    logger.info("=== Model Test ===")

    classifier = GRUClassifier(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES
    )

    # Sahte veri ile test
    fake_input = np.random.randn(config.MAX_SEQ_LENGTH, config.INPUT_SIZE).astype(np.float32)
    pred_class, confidence = classifier.predict(fake_input)

    logger.info(f"Tahmin sınıfı: {pred_class}")
    logger.info(f"Güven skoru: {confidence:.4f}")
    logger.info("Model testi başarılı! ✅")


def live_demo():
    """Kamera ile canlı demo (eğitilmiş model gerekli)."""
    config = Config()
    logger = Logger()
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        logger.error("Eğitilmiş model bulunamadı! Önce train.py çalıştır.")
        return

    logger.info("=== Canlı Demo ===")
    # TODO: Eğitim sonrası aktif edilecek


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="İşaret Dili Tanıma Sistemi")
    parser.add_argument('--mode', type=str, default='test_dataset',
                        choices=['test_dataset', 'test_model', 'live_demo'],
                        help='Çalıştırma modu')
    args = parser.parse_args()

    if args.mode == 'test_dataset':
        test_dataset()
    elif args.mode == 'test_model':
        test_model()
    elif args.mode == 'live_demo':
        live_demo()


