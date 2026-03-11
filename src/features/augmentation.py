"""
Data Augmentation modülü - İskelet (landmark) verileri için.

Görüntü augmentation'ından farklı olarak, burada 543 landmark noktasının
(x, y, z) koordinatları üzerinde dönüşümler yapılır.

Her augmentation bağımsızdır ve belirli bir olasılıkla uygulanır.
Eğitim sırasında on-the-fly (anlık) çalışır, böylece her epoch'ta
model verilerin farklı varyasyonlarını görür.
"""

import torch
import numpy as np


class LandmarkAugmentor:
    """
    İskelet landmark verileri için augmentation sınıfı.

    Desteklenen augmentasyonlar:
    1. Gaussian Noise    - Koordinatlara rastgele gürültü ekler
    2. Temporal Scale    - Videoyu hızlandırır/yavaşlatır
    3. Spatial Scale     - Koordinatları büyütür/küçültür
    4. Horizontal Flip   - Sol/sağ el landmark'larını değiştirir
    5. Frame Dropout     - Rastgele kareleri siler
    6. Temporal Shift    - Zamansal kaydırma uygular

    Kullanım:
        augmentor = LandmarkAugmentor()
        augmented_seq = augmentor(sequence)  # (seq_len, features) tensor
    """

    def __init__(self,
                 noise_prob=0.5, noise_std=0.01,
                 temporal_scale_prob=0.5, temporal_scale_range=(0.8, 1.2),
                 spatial_scale_prob=0.5, spatial_scale_range=(0.9, 1.1),
                 flip_prob=0.5,
                 frame_dropout_prob=0.3, frame_dropout_ratio=0.1,
                 temporal_shift_prob=0.3, shift_range=5,
                 max_seq_length=60):
        """
        Args:
            noise_prob: Gürültü ekleme olasılığı
            noise_std: Gürültü standart sapması (küçük tutulmalı)
            temporal_scale_prob: Zamansal ölçekleme olasılığı
            temporal_scale_range: Hız değişim aralığı (0.8=yavaş, 1.2=hızlı)
            spatial_scale_prob: Uzamsal ölçekleme olasılığı
            spatial_scale_range: Koordinat büyütme/küçültme aralığı
            flip_prob: Yatay aynalama olasılığı
            frame_dropout_prob: Kare silme olasılığı
            frame_dropout_ratio: Silinecek kare oranı
            temporal_shift_prob: Zamansal kaydırma olasılığı
            shift_range: Maksimum kaydırma miktarı (kare sayısı)
            max_seq_length: Hedef sekans uzunluğu (padding/truncate)
        """
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.temporal_scale_prob = temporal_scale_prob
        self.temporal_scale_range = temporal_scale_range
        self.spatial_scale_prob = spatial_scale_prob
        self.spatial_scale_range = spatial_scale_range
        self.flip_prob = flip_prob
        self.frame_dropout_prob = frame_dropout_prob
        self.frame_dropout_ratio = frame_dropout_ratio
        self.temporal_shift_prob = temporal_shift_prob
        self.shift_range = shift_range
        self.max_seq_length = max_seq_length

        # Landmark index aralıkları (her biri x,y,z = 3 değer)
        # face: 0-467 (468 landmark × 3 = 1404 değer, index 0-1403)
        # pose: 468-500 (33 landmark × 3 = 99 değer, index 1404-1502)
        # left_hand: 501-521 (21 landmark × 3 = 63 değer, index 1503-1565)
        # right_hand: 522-542 (21 landmark × 3 = 63 değer, index 1566-1628)
        self.face_start = 0
        self.face_end = 1404          # 468 * 3
        self.pose_start = 1404
        self.pose_end = 1503          # 1404 + 33 * 3
        self.left_hand_start = 1503
        self.left_hand_end = 1566     # 1503 + 21 * 3
        self.right_hand_start = 1566
        self.right_hand_end = 1629    # 1566 + 21 * 3

    def __call__(self, sequence):
        """
        Augmentation uygular.

        Args:
            sequence: (seq_len, features) boyutunda torch tensor

        Returns:
            Augmented (seq_len, features) tensor
        """
        # Tensor → numpy (daha kolay manipülasyon)
        if isinstance(sequence, torch.Tensor):
            seq = sequence.numpy().copy()
        else:
            seq = sequence.copy()

        # 1. Temporal Scale (zaman ölçekleme) — frame sayısı değişir
        if np.random.random() < self.temporal_scale_prob:
            seq = self._temporal_scale(seq)

        # 2. Frame Dropout (kare silme)
        if np.random.random() < self.frame_dropout_prob:
            seq = self._frame_dropout(seq)

        # 3. Temporal Shift (zamansal kaydırma)
        if np.random.random() < self.temporal_shift_prob:
            seq = self._temporal_shift(seq)

        # 4. Gaussian Noise (gürültü ekleme)
        if np.random.random() < self.noise_prob:
            seq = self._add_noise(seq)

        # 5. Spatial Scale (uzamsal ölçekleme)
        if np.random.random() < self.spatial_scale_prob:
            seq = self._spatial_scale(seq)

        # 6. Horizontal Flip (yatay aynalama)
        if np.random.random() < self.flip_prob:
            seq = self._horizontal_flip(seq)

        # Sekans uzunluğunu sabitle (padding/truncate)
        seq = self._fix_length(seq)

        return torch.FloatTensor(seq)

    def _add_noise(self, seq):
        """Koordinatlara küçük Gaussian gürültü ekler."""
        noise = np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)
        # Sıfır olan yerlere (padding) gürültü ekleme
        mask = np.any(seq != 0, axis=1, keepdims=True)
        noise = noise * mask
        return seq + noise

    def _temporal_scale(self, seq):
        """
        Videoyu hızlandırır veya yavaşlatır.
        0.8 = yavaşlatma (daha fazla kare), 1.2 = hızlandırma (daha az kare)
        """
        # Gerçek (sıfır olmayan) frame sayısını bul
        non_zero = np.any(seq != 0, axis=1)
        real_len = np.sum(non_zero)
        if real_len < 3:
            return seq

        real_seq = seq[non_zero]

        # Yeni uzunluk hesapla
        scale = np.random.uniform(*self.temporal_scale_range)
        new_len = max(3, int(real_len * scale))

        # Interpolasyon ile yeniden örnekle
        old_indices = np.linspace(0, real_len - 1, real_len)
        new_indices = np.linspace(0, real_len - 1, new_len)

        new_seq = np.zeros((new_len, seq.shape[1]), dtype=np.float32)
        for feat in range(seq.shape[1]):
            new_seq[:, feat] = np.interp(new_indices, old_indices, real_seq[:, feat])

        return new_seq

    def _spatial_scale(self, seq):
        """Tüm koordinatları rastgele bir faktörle ölçekler."""
        scale = np.random.uniform(*self.spatial_scale_range)
        mask = seq != 0
        seq = np.where(mask, seq * scale, seq)
        return seq

    def _horizontal_flip(self, seq):
        """
        Yatay aynalama: x koordinatlarını tersine çevirir ve
        sol el / sağ el landmark'larını birbirleriyle değiştirir.
        """
        result = seq.copy()

        # x koordinatlarını aynala (her 3. değer: x, y, z, x, y, z, ...)
        # x değerleri: 0, 3, 6, 9, ... indekslerde
        for i in range(0, seq.shape[1], 3):
            mask = result[:, i] != 0
            result[:, i] = np.where(mask, 1.0 - result[:, i], 0.0)

        # Sol el ve sağ el landmark'larını değiştir
        left_hand = result[:, self.left_hand_start:self.left_hand_end].copy()
        right_hand = result[:, self.right_hand_start:self.right_hand_end].copy()
        result[:, self.left_hand_start:self.left_hand_end] = right_hand
        result[:, self.right_hand_start:self.right_hand_end] = left_hand

        return result

    def _frame_dropout(self, seq):
        """Rastgele kareleri siler (modeli eksik veriye dayanıklı yapar)."""
        non_zero = np.any(seq != 0, axis=1)
        real_len = np.sum(non_zero)
        if real_len < 5:
            return seq

        real_seq = seq[non_zero]

        # Silinecek kare sayısı
        n_drop = max(1, int(real_len * self.frame_dropout_ratio))
        drop_indices = np.random.choice(real_len, n_drop, replace=False)

        # Kareleri sil
        keep_mask = np.ones(real_len, dtype=bool)
        keep_mask[drop_indices] = False
        new_seq = real_seq[keep_mask]

        return new_seq

    def _temporal_shift(self, seq):
        """Sekansı zamansal olarak kaydırır (baştan/sondan kırpar)."""
        non_zero = np.any(seq != 0, axis=1)
        real_len = np.sum(non_zero)
        if real_len < 5:
            return seq

        real_seq = seq[non_zero]
        shift = np.random.randint(-self.shift_range, self.shift_range + 1)

        if shift > 0:
            # İlk 'shift' kareyi kırp
            real_seq = real_seq[min(shift, real_len - 1):]
        elif shift < 0:
            # Son 'shift' kareyi kırp
            real_seq = real_seq[:max(1, real_len + shift)]

        return real_seq

    def _fix_length(self, seq):
        """Sekansı sabit uzunluğa getirir (truncate veya pad)."""
        target = self.max_seq_length

        if len(seq) >= target:
            # Truncate
            return seq[:target].astype(np.float32)
        else:
            # Pad (sıfırlarla doldur)
            padded = np.zeros((target, seq.shape[1]), dtype=np.float32)
            padded[:len(seq)] = seq
            return padded
