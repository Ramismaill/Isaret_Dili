import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import Config


def parquet_to_vector(path, max_seq_length=60):
    """Parquet dosyasını (max_seq_length, 1629) numpy dizisine çevirir."""
    df = pd.read_parquet(path)
    num_frames = df['frame'].nunique()

    frames = []
    for frame_idx in range(num_frames):
        frame_data = df[df['frame'] == frame_idx]
        vector = []

        for lm_type in ['face', 'pose', 'left_hand', 'right_hand']:
            type_data = frame_data[frame_data['type'] == lm_type]
            if len(type_data) > 0:
                type_data = type_data.sort_values('landmark_index')
                coords = type_data[['x', 'y', 'z']].values.flatten()
                vector.extend(coords)
            else:
                if lm_type == 'face':
                    vector.extend([0.0] * (468 * 3))
                elif lm_type == 'pose':
                    vector.extend([0.0] * (33 * 3))
                else:
                    vector.extend([0.0] * (21 * 3))

        frames.append(vector)

    sequence = np.array(frames, dtype=np.float32)
    sequence = np.nan_to_num(sequence, nan=0.0)

    if len(sequence) >= max_seq_length:
        start = (len(sequence) - max_seq_length) // 2
        sequence = sequence[start:start + max_seq_length]
    else:
        padded = np.zeros((max_seq_length, 1629), dtype=np.float32)
        padded[:len(sequence)] = sequence
        sequence = padded

    return sequence


def preprocess_split(split_name, config):
    """Checkpoint destekli ön işleme."""
    csv_path = os.path.join(config.DATASET_ROOT, f"{split_name}.csv")
    df = pd.read_csv(csv_path)

    save_dir = os.path.join(config.PROJECT_ROOT, "data", "processed")
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint dosyası
    checkpoint_path = os.path.join(save_dir, f"{split_name}_checkpoint.json")
    temp_dir = os.path.join(save_dir, f"{split_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Kaldığı yerden devam et
    start_idx = 0
    skipped = 0
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            cp = json.load(f)
            start_idx = cp['last_idx'] + 1
            skipped = cp['skipped']
        print(f"\n  ► Checkpoint bulundu! {start_idx}/{len(df)}'den devam ediliyor...")

    print(f"\n{'='*60}")
    print(f"{split_name.upper()} işleniyor — {len(df)} örnek (başlangıç: {start_idx})")
    print(f"{'='*60}")

    for idx in tqdm(range(start_idx, len(df)), desc=f"{split_name}",
                    initial=start_idx, total=len(df)):
        path = os.path.join(config.DATASET_ROOT, df.iloc[idx]['path'])
        label = df.iloc[idx]['sign']

        if not os.path.exists(path):
            skipped += 1
            # Checkpoint kaydet
            if idx % 500 == 0:
                with open(checkpoint_path, 'w') as f:
                    json.dump({'last_idx': idx, 'skipped': skipped}, f)
            continue

        try:
            sequence = parquet_to_vector(path, config.MAX_SEQ_LENGTH)

            # Her örneği ayrı kaydet (RAM tasarrufu)
            torch.save({
                'sequence': torch.FloatTensor(sequence),
                'label': label
            }, os.path.join(temp_dir, f"{idx}.pt"))

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [UYARI] Atlandı: {path} -> {e}")

        # Her 500 örnekte checkpoint kaydet
        if idx % 500 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({'last_idx': idx, 'skipped': skipped}, f)

    # Son checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump({'last_idx': len(df) - 1, 'skipped': skipped}, f)

    # Tüm parçaları birleştir
    print(f"\n  Parçalar birleştiriliyor...")
    all_sequences = []
    all_labels = []

    temp_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.pt')],
                        key=lambda x: int(x.split('.')[0]))

    for tf in tqdm(temp_files, desc="Birleştirme"):
        data = torch.load(os.path.join(temp_dir, tf), weights_only=False)
        all_sequences.append(data['sequence'])
        all_labels.append(data['label'])

    sequences_tensor = torch.stack(all_sequences)
    labels_tensor = torch.LongTensor(all_labels)

    # Final dosyayı kaydet
    save_path = os.path.join(save_dir, f"{split_name}.pt")
    torch.save({
        'sequences': sequences_tensor,
        'labels': labels_tensor
    }, save_path)

    file_size = os.path.getsize(save_path) / (1024**3)
    print(f"\n✅ {split_name}: {len(all_sequences)} örnek kaydedildi")
    print(f"   Boyut: {sequences_tensor.shape}")
    print(f"   Dosya: {save_path} ({file_size:.2f} GB)")
    print(f"   Atlanan: {skipped}")

    # Temp dosyaları temizle
    import shutil
    shutil.rmtree(temp_dir)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return len(all_sequences)


def main():
    config = Config()

    print("=" * 60)
    print("DATASET ÖN İŞLEME (Checkpoint Destekli)")
    print(f"Kaynak: {config.DATASET_ROOT}")
    print(f"Hedef: {config.PROJECT_ROOT}/data/processed/")
    print("Durdurup tekrar çalıştırabilirsin — kaldığı yerden devam eder!")
    print("=" * 60)

    total = 0
    for split in ['train', 'val', 'test']:
        # Zaten tamamlanmış mı kontrol et
        final_path = os.path.join(config.PROJECT_ROOT, "data", "processed", f"{split}.pt")
        if os.path.exists(final_path):
            print(f"\n  ✅ {split}.pt zaten var, atlanıyor!")
            data = torch.load(final_path, weights_only=False)
            total += len(data['labels'])
            continue

        total += preprocess_split(split, config)

    print(f"\n{'='*60}")
    print(f"TAMAMLANDI! Toplam {total} örnek işlendi.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
