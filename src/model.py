import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------
# Utilities: features & splits
# ---------------------------
def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0):
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(np.full(len(df), default), index=df.index, name=col)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures required features exist and are forward-looking-safe (no leakage).
    Computes:
      - log returns for price_a, price_b
      - momentum (if missing) using price_a
      - rsi, bb_width, stoch_k (if missing) minimal implementations
      - trend_slope (for compatibility with controller.py)
    """
    df = df.copy()
    for col in ["price_a", "price_b"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Log returns
    df["ret_a"] = np.log(df["price_a"]).diff().fillna(0.0)
    df["ret_b"] = np.log(df["price_b"]).diff().fillna(0.0)
    # Momentum (5-period pct change) if not supplied
    if "momentum" not in df.columns:
        df["momentum"] = df["price_a"].pct_change(5).fillna(0.0)
    # RSI (14)
    if "rsi" not in df.columns:
        delta = df["price_a"].diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        rs = (roll_up / (roll_down + 1e-9)).replace([np.inf, -np.inf], 0.0)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        df["rsi"] = df["rsi"].fillna(50.0)
    # Bollinger band width (20, 2)
    if "bb_width" not in df.columns:
        ma = df["price_a"].rolling(20).mean()
        sd = df["price_a"].rolling(20).std()
        upper = ma + 2 * sd
        lower = ma - 2 * sd
        df["bb_width"] = ((upper - lower) / (ma + 1e-9)).fillna(0.0)
    # Stochastic K (14)
    if "stoch_k" not in df.columns:
        low14 = df["price_a"].rolling(14).min()
        high14 = df["price_a"].rolling(14).max()
        df["stoch_k"] = 100.0 * (df["price_a"] - low14) / (high14 - low14 + 1e-9)
        df["stoch_k"] = df["stoch_k"].fillna(50.0)
    # Trend slope (20-period EMA pct change)
    if "trend_slope" not in df.columns:
        ma = df["price_a"].ewm(span=20).mean()
        df["trend_slope"] = ma.pct_change().fillna(0.0)
    # Sentiment if missing -> neutral 0.0
    if "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    # A light volatility proxy (for strategies)
    df["volatility"] = df["ret_a"].rolling(20).std().fillna(0.0)
    return df

def walk_forward_splits(n: int, train_frac: float = 0.8, n_folds: int = 1) -> List[Tuple[int, int]]:
    """
    Returns list of (train_end_idx, val_end_idx) pairs for simple walk-forward.
    If n_folds=1, it's a single split: [0:train_end) train, [train_end:val_end) val.
    """
    splits = []
    train_end = int(n * train_frac)
    val_end = n
    if n_folds <= 1:
        splits.append((train_end, val_end))
    else:
        fold_size = (n - train_end) // n_folds
        for k in range(n_folds):
            val_end_k = train_end + (k + 1) * fold_size if k < n_folds - 1 else n
            splits.append((train_end + k * fold_size, val_end_k))
    return splits

# ---------------------------
# Losses: quantile & BCE
# ---------------------------
class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.q = torch.tensor(quantiles).view(1, -1)  # shape [1, Q]

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: [B, 2, Q] (for 2 assets, Q quantiles)
        target: [B, 2] (true returns for 2 assets)
        """
        B, A, Q = preds.shape
        tq = target.unsqueeze(-1).expand(B, A, Q)  # [B,2,Q]
        diff = tq - preds
        q = self.q.to(preds.device).expand(B, A, Q)
        loss = torch.maximum(q * diff, (q - 1) * diff)  # pinball
        return loss.mean()

# ---------------------------
# Model: Causal Transformer
# ---------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def _causal_mask(T: int, device) -> torch.Tensor:
    # [T, T] with -inf above diagonal
    mask = torch.full((T, T), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class TransformerTradingModel(nn.Module):
    """
    Predicts next-step log returns for price_a & price_b.
    Heads:
      - Quantiles: Q10, Q50, Q90 for each asset -> [B, 2, 3]
      - Direction logits (P(up) via sigmoid) for each asset -> [B, 2]
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.quantiles = quantiles
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_quant = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * len(quantiles))  # 2 assets * Q quantiles
        )
        self.head_dir = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # 2 assets logits
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, F]
        returns:
          q: [B, 2, Q]
          dir_logits: [B, 2]
        """
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = self.pos(h)
        mask = _causal_mask(T, x.device)
        h = self.encoder(h, mask=mask)
        h_last = self.norm(h[:, -1, :])  # [B, D]
        quant = self.head_quant(h_last)  # [B, 2*Q]
        quant = quant.view(B, 2, len(self.quantiles))
        dir_logits = self.head_dir(h_last)  # [B, 2]
        return {"q": quant, "dir_logits": dir_logits}

# ---------------------------
# Wrapper: training & inference
# ---------------------------
@dataclass
class TrainConfig:
    seq_len: int = 60
    batch_size: int = 64
    epochs: int = 120
    patience: int = 15
    min_delta: float = 1e-4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_dir: float = 0.2  # weight for direction BCE in joint loss

class BetterTradingModel:
    """
    High-level API similar to your existing class:
      - fit_scaler(train_df)
      - prepare_data(df)
      - train_model(X, y, ...)
      - predict_next(recent_df) -> dict with quantiles/prices and direction probs
    """
    def __init__(self, feature_cols: Optional[List[str]] = None, device: Optional[torch.device] = None):
        self.feature_cols = feature_cols or ["price_a", "price_b", "sentiment", "momentum", "rsi", "bb_width", "stoch_k", "trend_slope"]
        self.scaler = StandardScaler()
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        self.model: Optional[TransformerTradingModel] = None
        self.cfg = TrainConfig()
        self.quantiles = [0.1, 0.5, 0.9]
        self.best_val = np.inf

    def fit_scaler(self, df: pd.DataFrame) -> None:
        df = build_features(df)
        X = df[self.feature_cols].values.astype(np.float32)
        self.scaler.fit(X)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        df = build_features(df)
        X = df[self.feature_cols].values.astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def make_sequences(self, df: pd.DataFrame, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          X: [N, T, F]
          y_ret: [N, 2] (next-step returns for price_a, price_b)
          y_dir: [N, 2] (0/1 labels for up move)
        """
        df = build_features(df)
        Xall = self._transform_features(df)
        ret = df[["ret_a", "ret_b"]].values.astype(np.float32)
        xs, yret, ydir = [], [], []
        for i in range(len(df) - seq_len - 1):
            xs.append(Xall[i:i+seq_len])
            next_ret = ret[i + seq_len]  # next-step return (1-step ahead)
            yret.append(next_ret)
            ydir.append((next_ret > 0).astype(np.float32))
        X = torch.tensor(np.array(xs), dtype=torch.float32)
        y_ret = torch.tensor(np.array(yret), dtype=torch.float32)
        y_dir = torch.tensor(np.array(ydir), dtype=torch.float32)
        return X, y_ret, y_dir

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        cfg: Optional[TrainConfig] = None
    ) -> None:
        cfg = cfg or self.cfg
        # fit scaler on *train only*
        self.fit_scaler(train_df)
        X_train, yret_train, ydir_train = self.make_sequences(train_df, cfg.seq_len)
        if val_df is not None:
            X_val, yret_val, ydir_val = self.make_sequences(val_df, cfg.seq_len)
        else:
            X_val = yret_val = ydir_val = None
        n_features = X_train.shape[-1]
        self.model = TransformerTradingModel(n_features=n_features, quantiles=self.quantiles).to(self.device)
        q_loss = QuantileLoss(self.quantiles)
        bce = nn.BCEWithLogitsLoss()
        opt = optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=8, min_lr=1e-5, verbose=False)
        tr_ds = TensorDataset(X_train, yret_train, ydir_train)
        tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
        if X_val is not None:
            va_ds = TensorDataset(X_val, yret_val, ydir_val)
            va_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)
        best = np.inf
        patience = 0
        for epoch in range(cfg.epochs):
            self.model.train()
            train_losses = []
            for xb, yb_ret, yb_dir in tr_dl:
                xb = xb.to(self.device)
                yb_ret = yb_ret.to(self.device)
                yb_dir = yb_dir.to(self.device)
                out = self.model(xb)
                loss_q = q_loss(out["q"], yb_ret)
                loss_dir = bce(out["dir_logits"], yb_dir)
                loss = loss_q + cfg.lambda_dir * loss_dir
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                train_losses.append(loss.item())
            # Validation
            if X_val is not None:
                self.model.eval()
                val_losses = []
                dir_hits = []
                with torch.no_grad():
                    for xb, yb_ret, yb_dir in va_dl:
                        xb = xb.to(self.device)
                        yb_ret = yb_ret.to(self.device)
                        yb_dir = yb_dir.to(self.device)
                        out = self.model(xb)
                        vq = q_loss(out["q"], yb_ret)
                        vd = bce(out["dir_logits"], yb_dir)
                        vloss = vq + cfg.lambda_dir * vd
                        val_losses.append(vloss.item())
                        probs = torch.sigmoid(out["dir_logits"])
                        preds = (probs > 0.5).float()
                        dir_hits.append((preds == yb_dir).float().mean().item())
                val_loss = float(np.mean(val_losses))
                dir_acc = float(np.mean(dir_hits))
                sched.step(val_loss)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Train {np.mean(train_losses):.5f} | Val {val_loss:.5f} | DirAcc {dir_acc:.3f}")
                if val_loss + cfg.min_delta < best:
                    best = val_loss
                    self.best_val = val_loss
                    patience = 0
                    # keep best weights
                    best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= cfg.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Train {np.mean(train_losses):.5f}")
        if X_val is not None and 'best_state' in locals():
            self.model.load_state_dict(best_state)

    def predict_next(self, recent_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        recent_df must contain at least `seq_len` rows.
        Returns dict with:
          - 'price_a' quantiles (q10, q50, q90)
          - 'price_b' quantiles (q10, q50, q90)
          - 'prob_up' for both assets
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train_model first.")
        df = build_features(recent_df)
        if len(df) < self.cfg.seq_len + 1:
            raise ValueError(f"Need at least {self.cfg.seq_len+1} rows, got {len(df)}.")
        Xall = self._transform_features(df)
        seq = Xall[-self.cfg.seq_len:]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1,T,F]
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            q = out["q"][0].cpu().numpy()  # [2, 3]
            probs = torch.sigmoid(out["dir_logits"])[0].cpu().numpy()  # [2]
        # Convert return quantiles to price quantiles
        last_a = float(df["price_a"].iloc[-1])
        last_b = float(df["price_b"].iloc[-1])
        def to_price(last, qret_row):
            # qret_row: [3] returns
            return last * np.exp(qret_row)
        qa = to_price(last_a, q[0])  # [Q]
        qb = to_price(last_b, q[1])  # [Q]
        return {
            "price_a_quantiles": qa,  # [q10, q50, q90]
            "price_b_quantiles": qb,
            "prob_up": probs,  # [p_up_a, p_up_b]
            "last_prices": np.array([last_a, last_b]),
        }