import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TradingModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, output_size=2, price_scaler=None, feature_scaler=None):
        super(TradingModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.price_scaler = price_scaler
        self.feature_scaler = feature_scaler
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-6)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.val_loss = 1e5  # Initialize to large finite value
    
    def forward(self, x):
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def train_model(self, X, y, X_val=None, y_val=None, epochs=150, patience=15, min_delta=0.0003, batch_size=32):
        print(f"Training input shape: {X.shape}, Target shape: {y.shape}")
        dataset = TensorDataset(X.to(self.device), y.to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            print(f"Validation input shape: {X_val.shape}, Target shape: {y_val.shape}")
        
        best_loss = 1e5
        counter = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {epoch_loss:.6f}')
            
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Val Loss: {val_loss:.6f}')
                
                self.scheduler.step(val_loss)
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    counter = 0
                    self.val_loss = val_loss
                else:
                    counter += 1
                    if counter >= patience:
                        print(f'Early stopping at epoch {epoch}')
                        break
                self.train()
    
    def predict_next(self, recent_data):
        features = ['price_a', 'price_b', 'sentiment', 'momentum', 'rsi', 'bb_width', 'stoch_k']
        price_data = recent_data[['price_a', 'price_b']].tail(60)
        feature_data = recent_data[['sentiment', 'momentum', 'rsi', 'bb_width', 'stoch_k']].tail(60)
        scaled_prices = self.price_scaler.transform(price_data)
        scaled_features = self.feature_scaler.transform(feature_data)
        scaled = np.hstack([scaled_prices, scaled_features])
        seq = torch.tensor(scaled.reshape(1, -1, self.input_size), dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            pred = self.forward(seq)[0]
        pred_prices = self.price_scaler.inverse_transform(pred.cpu().numpy().reshape(1, -1))[0]
        print(f"Raw prediction: {pred.cpu().numpy()}, Inverse prediction: {pred_prices}")
        return pred_prices
    
    def prepare_data(self, data, seq_length=60):
        features = ['price_a', 'price_b', 'sentiment', 'momentum', 'rsi', 'bb_width', 'stoch_k']
        price_data = data[['price_a', 'price_b']]
        feature_data = data[['sentiment', 'momentum', 'rsi', 'bb_width', 'stoch_k']]
        scaled_prices = self.price_scaler.transform(price_data)
        scaled_features = self.feature_scaler.transform(feature_data)
        scaled_data = np.hstack([scaled_prices, scaled_features])
        # Add small noise to prevent overfitting
        scaled_data += np.random.normal(0, 0.01, scaled_data.shape)
        xs, ys = [], []
        for i in range(len(scaled_data) - seq_length):
            xs.append(scaled_data[i:i+seq_length])
            ys.append(scaled_data[i+seq_length, :2])
        xs_tensor = torch.tensor(np.array(xs), dtype=torch.float32)
        ys_tensor = torch.tensor(np.array(ys), dtype=torch.float32)
        print(f"Prepared data shapes: X={xs_tensor.shape}, y={ys_tensor.shape}")
        return xs_tensor, ys_tensor

class EnsembleModel:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.model1 = TradingModel(hidden_size=256, price_scaler=self.price_scaler, feature_scaler=self.feature_scaler)
        self.model2 = TradingModel(hidden_size=64, price_scaler=self.price_scaler, feature_scaler=self.feature_scaler)
        self.model3 = TradingModel(hidden_size=32, price_scaler=self.price_scaler, feature_scaler=self.feature_scaler)
    
    def fit_scaler(self, data):
        price_features = ['price_a', 'price_b']
        other_features = ['sentiment', 'momentum', 'rsi', 'bb_width', 'stoch_k']
        self.price_scaler.fit(data[price_features])
        self.feature_scaler.fit(data[other_features])
        print(f"Price scaler range: min={self.price_scaler.data_min_}, max={self.price_scaler.data_max_}")
    
    def train_model(self, X, y, X_val=None, y_val=None, epochs=150, patience=15):
        self.model1.train_model(X, y, X_val, y_val, epochs, patience)
        self.model2.train_model(X, y, X_val, y_val, epochs, patience)
        self.model3.train_model(X, y, X_val, y_val, epochs, patience)
        print(f"Model losses: model1={self.model1.val_loss:.6f}, model2={self.model2.val_loss:.6f}, model3={self.model3.val_loss:.6f}")
    
    def predict_next(self, recent_data):
        pred1 = self.model1.predict_next(recent_data)
        pred2 = self.model2.predict_next(recent_data)
        pred3 = self.model3.predict_next(recent_data)
        losses = np.array([self.model1.val_loss, self.model2.val_loss, self.model3.val_loss])
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)) or np.sum(np.exp(-losses)) == 0:
            weights = np.array([1/3, 1/3, 1/3])  # Fallback to equal weights
        else:
            weights = np.exp(-losses) / np.sum(np.exp(-losses))
        w1, w2, w3 = weights
        print(f"Ensemble weights: w1={w1:.3f}, w2={w2:.3f}, w3={w3:.3f}")
        pred = w1 * pred1 + w2 * pred2 + w3 * pred3
        print(f"Ensemble prediction: {pred}")
        if np.any(np.isnan(pred)):
            print(f"Warning: NaN in ensemble prediction, returning last price_a: {recent_data['price_a'].iloc[-1]}")
            return np.array([recent_data['price_a'].iloc[-1], recent_data['price_b'].iloc[-1]])
        return pred