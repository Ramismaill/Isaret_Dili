import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Proje modülleri
import sys
sys.path.insert(0, os.path.dirname(__file__))

from configs.config import Config
from src.features.dataset_loader import create_dataloaders
from src.utils.helpers import Logger, ensure_dir


def create_model(config):
    """Config'e göre model oluşturur."""
    if config.MODEL_TYPE == "transformer":
        from src.models.transformer_classifier import SignLanguageTransformer
        model = SignLanguageTransformer(
            input_size=config.TRANSFORMER_INPUT_SIZE,
            num_classes=config.NUM_CLASSES,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FF,
            dropout=config.TRANSFORMER_DROPOUT,
            max_seq_length=config.MAX_SEQ_LENGTH + 1,  # +1 CLS token icin
            use_velocity=getattr(config, 'USE_VELOCITY', True)
        )
    else:
        from src.models.temporal_classifier import CNN_GRU
        model = CNN_GRU(
            input_size=config.CNN_GRU_INPUT_SIZE,
            num_classes=config.NUM_CLASSES,
            cnn_filters=config.CNN_FILTERS,
            cnn_kernel=config.CNN_KERNEL,
            gru_hidden=config.GRU_HIDDEN,
            gru_layers=config.GRU_LAYERS,
            dropout=config.GRU_DROPOUT
        )
    return model


class Trainer:
    """
    Mixed Precision destekli eğitim sınıfı.
    Transformer ve CNN+GRU modellerini destekler.
    """

    def __init__(self, config):
        self.config = config
        self.logger = Logger(config.LOG_DIR)
        self.device = torch.device(config.DEVICE)

        # Model
        self.model = create_model(config).to(self.device)
        self.logger.info(f"Model tipi: {config.MODEL_TYPE}")

        # Parametre sayısı
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parametreleri: {total_params:,}")

        # Loss, Optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # LR Scheduler: Warmup (5 epoch) + Cosine Annealing
        warmup_epochs = getattr(config, 'WARMUP_EPOCHS', 5)
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS - warmup_epochs, eta_min=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs]
        )

        # Mixed Precision
        self.scaler = GradScaler('cuda', enabled=config.USE_AMP)
        self.use_amp = config.USE_AMP

        # Early Stopping
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # Model kayıt dizini
        ensure_dir(config.MODEL_SAVE_DIR)

    def train_one_epoch(self, train_loader, epoch):
        """Bir epoch eğitim."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]",
                    ncols=100)

        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward
            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

            # Backward
            self.scaler.scale(loss).backward()

            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrikler
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Progress bar güncelle
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{acc:.2f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]",
                    ncols=100)

        for sequences, labels in pbar:
            sequences = sequences.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', enabled=self.use_amp):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Checkpoint kaydeder."""
        last_path = os.path.join(self.config.MODEL_SAVE_DIR, "last_checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'patience_counter': self.patience_counter,
            'model_type': self.config.MODEL_TYPE,
        }, last_path)

        if is_best:
            best_path = os.path.join(self.config.MODEL_SAVE_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model_type': self.config.MODEL_TYPE,
            }, best_path)
            self.logger.info(f"En iyi model kaydedildi: {best_path}")

    def load_checkpoint(self):
        """Kaydedilmiş checkpoint'tan devam eder."""
        path = os.path.join(self.config.MODEL_SAVE_DIR, "last_checkpoint.pth")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)

            # Model tipi uyuşuyor mu kontrol et
            saved_type = checkpoint.get('model_type', 'cnn_gru')
            if saved_type != self.config.MODEL_TYPE:
                self.logger.info(
                    f"Checkpoint model tipi ({saved_type}) mevcut tipten "
                    f"({self.config.MODEL_TYPE}) farklı — sıfırdan başlanıyor."
                )
                return 0

            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.best_val_acc = checkpoint['best_val_acc']
                self.patience_counter = checkpoint['patience_counter']
                start_epoch = checkpoint['epoch'] + 1
                self.logger.info(
                    f"Checkpoint yuklendi! Epoch {start_epoch}'dan devam ediliyor | "
                    f"En iyi Val Acc: {self.best_val_acc:.2f}%"
                )
                return start_epoch
            except (RuntimeError, KeyError) as e:
                self.logger.info(
                    f"Checkpoint mimarisi uyumsuz, sifirdan baslaniyorr: {e}"
                )
                return 0
        return 0

    def train(self, train_loader, val_loader):
        """Ana eğitim döngüsü."""
        self.logger.info("=" * 60)
        self.logger.info(f"EĞİTİM BAŞLIYOR — {self.config.MODEL_TYPE.upper()}")
        self.logger.info(f"Epochs: {self.config.EPOCHS} | "
                        f"Batch: {self.config.BATCH_SIZE} | "
                        f"LR: {self.config.LEARNING_RATE}")
        self.logger.info(f"AMP: {self.use_amp} | Device: {self.device}")
        self.logger.info("=" * 60)

        start_time = time.time()
        start_epoch = self.load_checkpoint()

        for epoch in range(start_epoch, self.config.EPOCHS):
            epoch_start = time.time()

            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            # LR scheduler step (epoch bazli)
            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            self.logger.info(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Süre: {epoch_time:.1f}s"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                self.logger.info(
                    f"[BEST] Yeni en iyi model! Val Acc: {val_acc:.2f}%"
                )
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
                self.logger.info(
                    f"Patience: {self.patience_counter}/{self.config.PATIENCE}"
                )

            if self.patience_counter >= self.config.PATIENCE:
                self.logger.info("Early stopping tetiklendi!")
                break

        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"EĞİTİM TAMAMLANDI | "
                        f"Toplam süre: {total_time/60:.1f} dk | "
                        f"En iyi Val Acc: {self.best_val_acc:.2f}%")
        self.logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true',
                        help='Sıfırdan eğitim (checkpoint yükleme)')
    parser.add_argument('--no-augment', action='store_true',
                        help='Augmentation olmadan eğit')
    args = parser.parse_args()

    config = Config()
    ensure_dir(config.MODEL_SAVE_DIR)

    augment = not args.no_augment
    print(f"Model: {config.MODEL_TYPE} | Augmentation: {'AÇIK' if augment else 'KAPALI'}")
    print("Dataset yükleniyor...")
    train_loader, val_loader = create_dataloaders(config, augment_train=augment)

    trainer = Trainer(config)

    if args.fresh:
        print("[!] Sıfırdan eğitim başlatılıyor")
        import shutil
        for fname in ["last_checkpoint.pth", "best_model.pth"]:
            fpath = os.path.join(config.MODEL_SAVE_DIR, fname)
            if os.path.exists(fpath):
                backup = fpath.replace(".pth", f"_backup_{config.MODEL_TYPE}.pth")
                shutil.copy2(fpath, backup)
                os.remove(fpath)
                print(f"  Yedeklendi: {backup}")

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
