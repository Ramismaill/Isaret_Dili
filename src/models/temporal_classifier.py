from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np


class TemporalSequenceClassifier(ABC):
    """
    Zamansal modelleme için soyut taban sınıf.
    """

    @abstractmethod
    def predict(self, sequence: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass


class CNN_GRU(nn.Module):
    """
    1D-CNN + GRU Hibrit Mimarisi.
    - 1D-CNN: Ardışık kareler arasındaki mikro hareketleri yakalar
    - GRU: Kelime düzeyindeki zamansal anlamı modeller
    """

    def __init__(self, input_size=1629, num_classes=226,
                 cnn_filters=64, cnn_kernel=5,
                 gru_hidden=128, gru_layers=2, dropout=0.3):
        super(CNN_GRU, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # 1D-CNN katmanları
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size,
                      out_channels=cnn_filters,
                      kernel_size=cnn_kernel,
                      padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(in_channels=cnn_filters,
                      out_channels=cnn_filters * 2,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GRU katmanı
        self.gru = nn.GRU(
            input_size=cnn_filters * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False
        )

        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, cnn_filters*2)
        x, _ = self.gru(x)
        x = x[:, -1, :]         # Son zaman adımı
        x = self.classifier(x)
        return x


class GRUClassifier(TemporalSequenceClassifier):
    """
    CNN_GRU modelini saran wrapper sınıf.
    """

    def __init__(self, input_size=1629, num_classes=226, model_path=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"[GRUClassifier] Cihaz: {self.device}")

        self.model = CNN_GRU(
            input_size=input_size,
            num_classes=num_classes
        ).to(self.device)

        if model_path:
            self.load_model(model_path)

        # Parametre sayısını göster
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[GRUClassifier] Toplam parametre: {total_params:,} | "
              f"Eğitilebilir: {trainable:,}")

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
        print(f"[GRUClassifier] Model yüklendi: {path}")

    def save_model(self, path: str, extra_info: dict = None):
        save_dict = {'model_state_dict': self.model.state_dict()}
        if extra_info:
            save_dict.update(extra_info)
        torch.save(save_dict, path)
        print(f"[GRUClassifier] Model kaydedildi: {path}")