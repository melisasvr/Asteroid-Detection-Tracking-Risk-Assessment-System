# ml/models.py
"""
Machine Learning Models for AsteroidWatch
==========================================
Two complementary models:

1. CNN STREAK CLASSIFIER
   Input  : 64×64 image cutout around a candidate detection
   Output : P(streak | image) ∈ [0, 1]
   Purpose: Re-score CV detections, reduce false positives from cosmic rays / stars

2. LSTM TRAJECTORY FORECASTER
   Input  : sequence of (x, y, z) positions over time (in AU, ecliptic frame)
   Output : predicted future positions + uncertainty
   Purpose: Short-term trajectory forecasting between observational arcs

ARCHITECTURE NOTES
------------------
CNN:
  Lightweight MobileNet-inspired design (depthwise separable convolutions).
  Works on 64×64 crops — fast inference, no GPU required for portfolio demo.
  In production, you'd use a ResNet-18 pre-trained on astronomical survey data.

LSTM:
  2-layer bidirectional LSTM → position encoder → future position decoder.
  Bidirectional: exploits both past AND future context during training
  (we know the full observed arc; bidirectional LSTM uses all of it).
  At inference time, we use the forward-only direction for true prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CNN STREAK CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution = depthwise + pointwise.
    3× fewer parameters than a standard conv — critical for real-time inference.
    MobileNet's key innovation (Howard et al. 2017).
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)   # per-channel
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)  # 1×1 mix channels
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


class StreakCNN(nn.Module):
    """
    Lightweight CNN classifier for 64×64 grayscale image patches.

    Architecture:
      Conv(1→16) → DSConv(16→32, s=2) → DSConv(32→64, s=2) →
      DSConv(64→128, s=2) → GlobalAvgPool → FC(128→64) → FC(64→1) → Sigmoid

    Input  : (batch, 1, 64, 64) float32, values normalised [0, 1]
    Output : (batch, 1) float32 ∈ [0, 1] — P(asteroid streak)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.body = nn.Sequential(
            DepthwiseSeparableConv(16,  32, stride=2),  # 64→32
            DepthwiseSeparableConv(32,  64, stride=2),  # 32→16
            DepthwiseSeparableConv(64, 128, stride=2),  # 16→8
            DepthwiseSeparableConv(128, 128, stride=2), #  8→4
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # Global average pool → (batch, 128, 1, 1)
            nn.Flatten(),              # → (batch, 128)
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.stem(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM TRAJECTORY FORECASTER
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryLSTM(nn.Module):
    """
    Sequence-to-sequence LSTM for asteroid trajectory prediction.

    Input  : (batch, seq_len, 6) — [x, y, z, vx, vy, vz] per timestep
             positions in AU, velocities in AU/day (ecliptic frame)
    Output : (batch, predict_steps, 3) — predicted [x, y, z] positions

    Architecture:
      Feature projection (6→hidden_dim) → Bidirectional LSTM (encode) →
      Linear bridge → Unidirectional LSTM (decode) → Output projection

    The encoder is BIDIRECTIONAL because during training we have the full arc.
    The decoder is UNIDIRECTIONAL because it's predicting the future.

    Uncertainty estimation via MC Dropout:
      At inference time, run N forward passes WITH dropout enabled.
      The variance across passes is a proxy for prediction uncertainty.
      This is Bayesian deep learning the easy way (Gal & Ghahramani 2016).
    """

    def __init__(self,
                 input_dim:    int = 6,
                 hidden_dim:   int = 64,
                 num_layers:   int = 2,
                 predict_steps: int = 5,
                 dropout:      float = 0.2):
        super().__init__()

        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.predict_steps = predict_steps

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Encoder: Bidirectional LSTM processes observed arc
        self.encoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        # Bridge: compress bidirectional hidden state → decoder init
        # Bidirectional doubles the hidden size → need to halve it
        self.bridge_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bridge_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # Decoder: Unidirectional LSTM predicts future
        self.decoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0.0,
        )

        # Output: project hidden state → 3 position coordinates
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode observed sequence.
        x: (batch, seq_len, input_dim)
        Returns (h, c) for decoder initialisation.
        """
        proj = self.input_proj(x)                           # (B, T, H)
        _, (h, c) = self.encoder(proj)                      # h: (2*layers, B, H)

        # Take last layer's forward+backward hidden states, concatenate
        # h has shape (num_layers * 2, batch, hidden) for bidirectional
        # We want last layer: h[-2] (forward), h[-1] (backward)
        h_fwd = h[-2]   # (B, H)
        h_bwd = h[-1]   # (B, H)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)   # (B, 2H)

        c_fwd = c[-2]
        c_bwd = c[-1]
        c_cat = torch.cat([c_fwd, c_bwd], dim=-1)

        # Bridge to decoder hidden size
        h_dec = torch.tanh(self.bridge_h(h_cat)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_dec = torch.tanh(self.bridge_c(c_cat)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h_dec, c_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass for training.
        x: (batch, seq_len, input_dim)
        Returns predicted positions: (batch, predict_steps, 3)
        """
        batch_size = x.size(0)
        h, c = self.encode(x)

        # Decoder input: start with last observed position (projected)
        last_obs = self.input_proj(x[:, -1:, :])  # (B, 1, H)

        outputs = []
        dec_input = last_obs

        for step in range(self.predict_steps):
            out, (h, c) = self.decoder(dec_input, (h, c))  # out: (B, 1, H)
            pos = self.output_proj(out)                      # (B, 1, 3)
            outputs.append(pos)
            # Autoregressive: feed output back as next input
            # Re-project the 3D position back to hidden_dim
            # Simple approach: zero-pad to input_dim
            padded = torch.zeros(batch_size, 1, 6, device=x.device)
            padded[:, :, :3] = pos
            dec_input = self.input_proj(padded)

        return torch.cat(outputs, dim=1)   # (B, predict_steps, 3)

    @torch.no_grad()
    def predict_with_uncertainty(self,
                                  x: torch.Tensor,
                                  n_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MC Dropout uncertainty estimation.
        Run N stochastic forward passes with dropout active.
        Returns (mean_prediction, std_prediction), both shape (predict_steps, 3).
        """
        self.train()   # Enable dropout (required for MC Dropout)

        preds = []
        for _ in range(n_samples):
            pred = self.forward(x)   # (1, steps, 3)
            preds.append(pred.cpu().numpy())

        self.eval()
        preds = np.stack(preds, axis=0)   # (n_samples, 1, steps, 3)
        mean  = preds.mean(axis=0)[0]     # (steps, 3)
        std   = preds.std(axis=0)[0]      # (steps, 3)
        return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Dataset of (observed_arc, future_positions) pairs for LSTM training.
    Generated from synthetic orbital mechanics data.
    """

    def __init__(self,
                 n_samples:     int   = 1000,
                 seq_len:       int   = 10,
                 predict_steps: int   = 5,
                 dt_days:       float = 1.0,
                 seed:          int   = 42):
        self.seq_len       = seq_len
        self.predict_steps = predict_steps
        self.data = self._generate(n_samples, seq_len, predict_steps, dt_days, seed)

    def _generate(self, n, seq_len, predict_steps, dt, seed):
        """Generate synthetic orbital trajectory sequences."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from orbits.orbit_engine import OrbitalElements, KeplerSolver

        rng    = np.random.default_rng(seed)
        solver = KeplerSolver()
        data   = []

        for _ in range(n):
            # Random NEO-like orbit
            a    = rng.uniform(0.5, 2.5)
            e    = rng.uniform(0.01, 0.6)
            i    = rng.uniform(0, 30)
            raan = rng.uniform(0, 360)
            argp = rng.uniform(0, 360)
            M0   = rng.uniform(0, 360)

            elems = OrbitalElements(
                object_id='train', a=a, e=e, i=i,
                raan=raan, argp=argp, M0=M0, epoch_jd=2451545.0
            )

            total_steps = seq_len + predict_steps
            # Random starting time
            t0 = rng.uniform(0, elems.period_yr * 365.25)

            positions = []
            velocities = []
            for step in range(total_steps):
                pos, vel = solver.elements_to_cartesian(elems, t0 + step * dt)
                positions.append(pos)
                velocities.append(vel)

            positions  = np.array(positions,  dtype=np.float32)
            velocities = np.array(velocities, dtype=np.float32)

            # Input: [x, y, z, vx, vy, vz] for each observed timestep
            obs_seq  = np.concatenate([positions[:seq_len],
                                        velocities[:seq_len]], axis=-1)
            # Target: [x, y, z] for each future timestep
            fut_pos  = positions[seq_len:]

            data.append((obs_seq, fut_pos))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, fut = self.data[idx]
        return torch.tensor(obs), torch.tensor(fut)


class StreakCutoutDataset(Dataset):
    """
    Dataset of 64×64 image cutouts labelled as streak (1) or non-streak (0).
    Generated from the simulator — real streaks + random background patches as negatives.
    """

    def __init__(self, n_images: int = 200, cutout_size: int = 64, seed: int = 0):
        self.cutout_size = cutout_size
        self.samples = self._generate(n_images, seed)

    def _generate(self, n_images, seed):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from data.simulator import TelescopeImageSimulator, SimulationConfig
        from detection.streak_detector import StreakDetector
        from utils.helpers import normalize_image

        samples = []
        cs = self.cutout_size // 2

        for i in range(n_images):
            cfg   = SimulationConfig(random_seed=seed + i, n_asteroids=3, n_cosmic_rays=5)
            sim   = TelescopeImageSimulator(cfg)
            image, truths = sim.generate()
            norm_image = normalize_image(image)
            H, W = image.shape

            # Positive samples: cutouts around true asteroid streaks
            for truth in truths:
                mx = int((truth.x_start + truth.x_end) / 2)
                my = int((truth.y_start + truth.y_end) / 2)
                r0 = np.clip(my - cs, 0, H - self.cutout_size)
                c0 = np.clip(mx - cs, 0, W - self.cutout_size)
                patch = norm_image[r0:r0+self.cutout_size, c0:c0+self.cutout_size]
                if patch.shape == (self.cutout_size, self.cutout_size):
                    samples.append((patch.copy(), 1))

            # Negative samples: random background patches (should have no streaks)
            rng = np.random.default_rng(seed + i + 10000)
            for _ in range(len(truths) * 2):
                ry = rng.integers(0, H - self.cutout_size)
                rx = rng.integers(0, W - self.cutout_size)
                patch = norm_image[ry:ry+self.cutout_size, rx:rx+self.cutout_size]
                samples.append((patch.copy(), 0))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        # (H, W) → (1, H, W) for CNN
        x = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_cnn(n_images: int = 100,
               epochs:   int = 20,
               lr:       float = 1e-3,
               save_path: str = "data/processed/streak_cnn.pt") -> StreakCNN:
    """Train the CNN streak classifier. Returns trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training CNN on {device}")
    print(f"Training streak CNN  (device={device}, epochs={epochs})")

    dataset    = StreakCutoutDataset(n_images=n_images)
    n_train    = int(0.8 * len(dataset))
    n_val      = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    model = StreakCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCELoss()

    history = []
    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).squeeze(-1)
                val_loss += loss_fn(pred, y).item()
                correct  += ((pred > 0.5).float() == y).sum().item()
                total    += y.size(0)

        sched.step()
        acc = correct / total
        tl  = train_loss / len(train_dl)
        vl  = val_loss   / len(val_dl)
        history.append({'epoch': epoch+1, 'train_loss': tl, 'val_loss': vl, 'val_acc': acc})

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={tl:.4f}  "
                  f"val_loss={vl:.4f}  val_acc={acc:.3f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'history': history}, save_path)
    print(f"CNN saved → {save_path}")
    return model


def train_lstm(n_samples:  int   = 2000,
               epochs:     int   = 30,
               lr:         float = 1e-3,
               seq_len:    int   = 10,
               pred_steps: int   = 5,
               save_path:  str   = "data/processed/trajectory_lstm.pt") -> TrajectoryLSTM:
    """Train the trajectory LSTM. Returns trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training trajectory LSTM  (device={device}, epochs={epochs})")

    dataset  = TrajectoryDataset(n_samples=n_samples, seq_len=seq_len,
                                  predict_steps=pred_steps)
    n_train  = int(0.85 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    model   = TrajectoryLSTM(hidden_dim=64, num_layers=2,
                              predict_steps=pred_steps).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for obs, fut in train_dl:
            obs, fut = obs.to(device), fut.to(device)
            pred = model(obs)
            loss = loss_fn(pred, fut)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, fut in val_dl:
                obs, fut = obs.to(device), fut.to(device)
                pred = model(obs)
                val_loss += loss_fn(pred, fut).item()

        tl = train_loss / len(train_dl)
        vl = val_loss   / len(val_dl)
        sched.step(vl)
        history.append({'epoch': epoch+1, 'train_loss': tl, 'val_loss': vl})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  train_loss={tl:.6f}  val_loss={vl:.6f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'history': history,
                'seq_len': seq_len, 'pred_steps': pred_steps}, save_path)
    print(f"LSTM saved → {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (for inference in the dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def load_cnn(path: str, device: str = 'cpu') -> Optional[StreakCNN]:
    """Load a trained CNN from disk. Returns None if file doesn't exist."""
    if not Path(path).exists():
        return None
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = StreakCNN()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def load_lstm(path: str, device: str = 'cpu') -> Optional[TrajectoryLSTM]:
    """Load a trained LSTM from disk."""
    if not Path(path).exists():
        return None
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = TrajectoryLSTM(
        predict_steps = ckpt.get('pred_steps', 5)
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== StreakCNN architecture ===")
    cnn = StreakCNN()
    dummy = torch.randn(4, 1, 64, 64)
    out   = cnn(dummy)
    print(f"  Input: {dummy.shape}  →  Output: {out.shape}")
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\n=== TrajectoryLSTM architecture ===")
    lstm  = TrajectoryLSTM(input_dim=6, hidden_dim=64, num_layers=2, predict_steps=5)
    dummy = torch.randn(4, 10, 6)
    out   = lstm(dummy)
    print(f"  Input: {dummy.shape}  →  Output: {out.shape}")
    n_params = sum(p.numel() for p in lstm.parameters())
    print(f"  Parameters: {n_params:,}")

    print("\n=== Training CNN (quick test: 5 images, 3 epochs) ===")
    model = train_cnn(n_images=5, epochs=3, save_path="/tmp/test_cnn.pt")
    print("CNN training complete.")

    print("\n=== Training LSTM (quick test: 100 samples, 3 epochs) ===")
    model = train_lstm(n_samples=100, epochs=3, save_path="/tmp/test_lstm.pt")
    print("LSTM training complete.")