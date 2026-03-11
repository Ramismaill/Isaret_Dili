import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.features.augmentation import LandmarkAugmentor


class AUTSLDataset(Dataset):
    """
    İşlenmiş .pt dosyasından okuyan Dataset sınıfı.
    Transformer modeli için yüz landmark'larını çıkarır,
    sadece pose + eller kullanır (225 özellik).
    """

    def __init__(self, pt_path, augment=False, max_seq_length=60,
                 use_face=False):
        data = torch.load(pt_path, weights_only=False)
        self.sequences = data['sequences']
        self.labels = data['labels']
        self.augment = augment
        self.use_face = use_face

        # Landmark index aralıkları
        self.pose_start = 1404       # 468 * 3
        self.pose_end = 1503         # 1404 + 33 * 3
        self.left_hand_start = 1503
        self.left_hand_end = 1566    # 1503 + 21 * 3
        self.right_hand_start = 1566
        self.right_hand_end = 1629   # 1566 + 21 * 3

        # Augmentor (hafif parametreler)
        if augment:
            # Augmentor input_size'ı use_face'e göre ayarla
            aug_features = 1629 if use_face else 225
            self.augmentor = LandmarkAugmentor(
                noise_prob=0.3, noise_std=0.005,
                temporal_scale_prob=0.2, temporal_scale_range=(0.9, 1.1),
                spatial_scale_prob=0.2, spatial_scale_range=(0.95, 1.05),
                flip_prob=0.0,
                frame_dropout_prob=0.15, frame_dropout_ratio=0.05,
                temporal_shift_prob=0.15, shift_range=3,
                max_seq_length=max_seq_length
            )

        mode = "AUGMENTED" if augment else "NORMAL"
        feat_mode = "TÜM (1629)" if use_face else "EL+POSE (225)"
        print(f"[AUTSLDataset] {len(self.labels)} örnek yüklendi | "
              f"Özellik: {feat_mode} | "
              f"Sınıf: {self.labels.unique().shape[0]} | "
              f"Mod: {mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Yüz landmark'larını çıkar (sadece pose + eller)
        if not self.use_face:
            pose = sequence[:, self.pose_start:self.pose_end]         # (60, 99)
            left_hand = sequence[:, self.left_hand_start:self.left_hand_end]  # (60, 63)
            right_hand = sequence[:, self.right_hand_start:self.right_hand_end]  # (60, 63)
            sequence = torch.cat([pose, left_hand, right_hand], dim=1)  # (60, 225)

        # Augmentation
        if self.augment:
            sequence = self.augmentor(sequence)

        return sequence, label


def create_dataloaders(config, augment_train=True):
    """Train ve validation DataLoader oluşturur."""
    processed_dir = os.path.join(config.PROJECT_ROOT, "data", "processed")

    # Transformer ise yüz kullanma, CNN+GRU ise yüz kullan
    use_face = (config.MODEL_TYPE == "cnn_gru")

    train_dataset = AUTSLDataset(
        os.path.join(processed_dir, "train.pt"),
        augment=augment_train,
        max_seq_length=config.MAX_SEQ_LENGTH,
        use_face=use_face
    )
    val_dataset = AUTSLDataset(
        os.path.join(processed_dir, "val.pt"),
        augment=False,
        use_face=use_face
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    aug_status = "AÇIK" if augment_train else "KAPALI"
    print(f"[DataLoader] Train: {len(train_loader)} batch | "
          f"Val: {len(val_loader)} batch | "
          f"Batch: {config.BATCH_SIZE} | "
          f"Aug: {aug_status} | "
          f"Model: {config.MODEL_TYPE}")

    return train_loader, val_loader
