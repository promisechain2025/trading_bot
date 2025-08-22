import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class TradingModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, num_layers=3, output_size=2, scaler=None):
        super(TradingModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_size, output_size)
        self.scaler = scaler  # Shared scaler passed from EnsembleModel
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.00015, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.val_loss = float('inf')
    
    def forward(self, x):
        x = x.to(self.device)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
    def train_model(self, X, y, X_val=None, y_val=None, epochs=100, patience=10, min_delta=0.0005):
        X = X.to(self.device)
        y = y.to(self.device)
        if X_val is not None and y_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        
        best_loss = float('inf')
        counter = 0
        
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            outputs = self.forward(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {loss.item()}')
            
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X_val)
                    val_loss = self.criterion(val_outputs, y_val).item()
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Val Loss: {val_loss}')
                
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
        features = ['price_a', 'price_b', 'sentiment', 'momentum', 'rsi', 'macd', 'bb_width', 'stoch_k', 'obv']
        scaled = self.scaler.transform(recent_data[features])
        seq = torch.tensor(scaled[-30:].reshape(1, -1, self.input_size), dtype=torch.float32)
        with torch.no_grad():
            pred = self.forward(seq)[0]
        inverse_input = [pred[0].item(), pred[1].item()] + [0] * (self.input_size - 2)
        return self.scaler.inverse_transform([inverse_input])[0][:2]
    
    def prepare_data(self, data, seq_length=30):
        features = ['price_a', 'price_b', 'sentiment', 'momentum', 'rsi', 'macd', 'bb_width', 'stoch_k', 'obv']
        scaled_data = self.scaler.fit_transform(data[features])
        xs, ys = [], []
        for i in range(len(scaled_data) - seq_length):
            xs.append(scaled_data[i:i+seq_length])
            ys.append(scaled_data[i+seq_length, :2])
        return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

class EnsembleModel:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model1 = TradingModel(hidden_size=256, scaler=self.scaler)
        self.model2 = TradingModel(hidden_size=128, scaler=self.scaler)
        self.model3 = TradingModel(hidden_size=64, scaler=self.scaler)
    
    def fit_scaler(self, data):
        features = ['price_a', 'price_b', 'sentiment', 'momentum', 'rsi', 'macd', 'bb_width', 'stoch_k', 'obv']
        self.scaler.fit(data[features])
    
    def train_model(self, X, y, X_val=None, y_val=None, epochs=100, patience=10):
        self.model1.train_model(X, y, X_val, y_val, epochs, patience)
        self.model2.train_model(X, y, X_val, y_val, epochs, patience)
        self.model3.train_model(X, y, X_val, y_val, epochs, patience)
    
    def predict_next(self, recent_data):
        pred1 = self.model1.predict_next(recent_data)
        pred2 = self.model2.predict_next(recent_data)
        pred3 = self.model3.predict_next(recent_data)
        total_loss = self.model1.val_loss + self.model2.val_loss + self.model3.val_loss
        w1 = (total_loss - self.model1.val_loss) / (total_loss * 2) if total_loss > 0 else 1/3
        w2 = (total_loss - self.model2.val_loss) / (total_loss * 2) if total_loss > 0 else 1/3
        w3 = (total_loss - self.model3.val_loss) / (total_loss * 2) if total_loss > 0 else 1/3
        return (w1 * pred1 + w2 * pred2 + w3 * pred3) / (w1 + w2 + w3)