"""
TSLFormer-inspired Transformer modeli v2.
MediaPipe landmark verileri ile Turk Isaret Dili tanima.

v2 Iyilestirmeler:
- Padding mask: Sifir frame'ler transformer tarafindan ignore edilir
- Velocity ozellikleri: Hareket bilgisi otomatik hesaplanir (pozisyon + hiz)
- Daha derin mimari (4 layer) + final LayerNorm
- Xavier weight initialization
"""

import math
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """
    Isaret Dili Tanima icin Transformer modeli v2.

    Girdi: (batch, seq_len, input_size) - landmark koordinatlari
    Cikti: (batch, num_classes) - sinif olasikliklari

    Yeni ozellikler:
    - Padding mask: Sifir frame'ler attention'da ignore edilir
    - Velocity: Frame arasi hiz bilgisi otomatik hesaplanir
    - Final LayerNorm: Transformer encoder ciktisinda normalizasyon
    """

    def __init__(self, input_size=225, num_classes=226,
                 d_model=512, nhead=8, num_layers=4,
                 dim_feedforward=2048, dropout=0.25,
                 max_seq_length=61, use_velocity=True):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.use_velocity = use_velocity

        # Velocity eklenirse input boyutu 2x olur (pozisyon + hiz)
        actual_input = input_size * 2 if use_velocity else input_size

        # Girdi projeksiyonu: actual_input -> d_model
        self.input_projection = nn.Sequential(
            nn.Linear(actual_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer Encoder (final LayerNorm ile)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # CLS token (siniflandirma icin ozel token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Siniflandirici
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Linear ve LayerNorm katmanlari icin uygun initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _compute_velocity(self, x):
        """
        Frame arasi hiz hesapla: v[t] = x[t] - x[t-1].
        Ilk frame'in hizi sifir olarak ayarlanir.
        """
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return velocity

    def _create_padding_mask(self, x):
        """
        Sifir frame'ler icin padding mask olusturur.
        True = bu pozisyon IGNORE edilecek (maskelenen)
        CLS token asla maskelenmez.
        """
        is_padding = (x.abs().sum(dim=-1) == 0)  # (batch, seq_len)
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([cls_mask, is_padding], dim=1)  # (batch, seq_len+1)
        return mask

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Padding mask olustur (orijinal input'tan, projeksiyon oncesi)
        padding_mask = self._create_padding_mask(x)

        # Velocity hesapla ve birlestir
        if self.use_velocity:
            velocity = self._compute_velocity(x)
            x = torch.cat([x, velocity], dim=-1)  # (batch, seq_len, input_size*2)

        # Girdi projeksiyonu
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # CLS token ekle
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder (padding mask ile)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # CLS token'in ciktisini al (ilk pozisyon)
        cls_output = x[:, 0, :]

        # Siniflandirma
        return self.classifier(cls_output)


class TransformerClassifier:
    """
    SignLanguageTransformer modelini saran wrapper sinif.
    Inference (canli demo) icin kullanilir.
    """

    def __init__(self, input_size=225, num_classes=226, model_path=None,
                 d_model=512, nhead=8, num_layers=4, use_velocity=True):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"[TransformerClassifier] Cihaz: {self.device}")

        self.model = SignLanguageTransformer(
            input_size=input_size,
            num_classes=num_classes,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            use_velocity=use_velocity
        ).to(self.device)

        if model_path:
            self.load_model(model_path)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TransformerClassifier] Toplam parametre: {total_params:,} | "
              f"Egitilebilir: {trainable:,}")

    def predict(self, sequence: np.ndarray) -> tuple:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            return predicted.item(), confidence.item()

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[TransformerClassifier] Model yuklendi: {path}")

    def save_model(self, path: str, extra_info: dict = None):
        save_dict = {'model_state_dict': self.model.state_dict()}
        if extra_info:
            save_dict.update(extra_info)
        torch.save(save_dict, path)
        print(f"[TransformerClassifier] Model kaydedildi: {path}")
