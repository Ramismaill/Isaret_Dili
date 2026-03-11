import os


class Config:
    """Tüm proje ayarları."""

    # === Dataset Yolları ===
    DATASET_ROOT = r"D:\autsl_dataset\austl-processed-dataset-mediapipe-landmarks\AUSTL_processed_landmark"
    TRAIN_CSV = os.path.join(DATASET_ROOT, "train.csv")
    VAL_CSV = os.path.join(DATASET_ROOT, "val.csv")
    TEST_CSV = os.path.join(DATASET_ROOT, "test.csv")

    # === Model Parametreleri ===
    NUM_CLASSES = 226
    MAX_SEQ_LENGTH = 60      # Maksimum frame sayısı (padding/truncate)

    # Model tipi: "transformer" veya "cnn_gru"
    MODEL_TYPE = "transformer"

    # --- Transformer parametreleri (TSLFormer-inspired v2) ---
    TRANSFORMER_INPUT_SIZE = 225   # pose(33x3) + left_hand(21x3) + right_hand(21x3)
    TRANSFORMER_D_MODEL = 512      # Hidden boyutu
    TRANSFORMER_NHEAD = 8          # Attention head sayisi
    TRANSFORMER_LAYERS = 4         # Encoder layer sayisi (3->4)
    TRANSFORMER_DIM_FF = 2048      # Feedforward boyutu
    TRANSFORMER_DROPOUT = 0.25
    USE_VELOCITY = True            # Velocity ozellikleri (hareket bilgisi)
    WARMUP_EPOCHS = 5              # LR warmup epoch sayisi

    # --- CNN+GRU parametreleri (eski model, yedek) ---
    CNN_GRU_INPUT_SIZE = 1629      # 543 landmark × 3 (tüm landmark'lar)
    CNN_FILTERS = 64
    CNN_KERNEL = 5
    GRU_HIDDEN = 128
    GRU_LAYERS = 2
    GRU_DROPOUT = 0.3

    # Landmark index aralıkları (1629 özellikli vektörde)
    FACE_START = 0
    FACE_END = 1404          # 468 × 3
    POSE_START = 1404
    POSE_END = 1503          # 1404 + 33 × 3
    LEFT_HAND_START = 1503
    LEFT_HAND_END = 1566     # 1503 + 21 × 3
    RIGHT_HAND_START = 1566
    RIGHT_HAND_END = 1629    # 1566 + 21 × 3

    # === Eğitim Parametreleri ===
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.0001       # Transformer için 1e-4 (TSLFormer'dan)
    WEIGHT_DECAY = 5e-4
    PATIENCE = 20                # Transformer daha yavaş öğrenir, sabır artırıldı

    # Mixed Precision
    USE_AMP = True               # Automatic Mixed Precision

    # === Proje Yolları ===
    PROJECT_ROOT = r"C:\Users\Muhammet\PycharmProjects\isaret_dili"
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    # === Cihaz ===
    DEVICE = "cuda"              # cuda veya cpu
