import torch
import torch.nn as nn
import torch.optim as optim
import sqlite3
import pandas as pd
import numpy as np
import logging
import os
from .nexus_transformer import TimeSeriesTransformer

logger = logging.getLogger("NexusTrainer")

class NexusTrainer:
    """
    The 'Gym' for the Transformer Brain.
    Extracts data from MarketMemory (SQLite), preprocesses it, and trains the model.
    """
    def __init__(self, db_path="data/market_memory.db", model_save_path="models/nexus_transformer.pth"):
        self.db_path = db_path
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.seq_len = 64
        self.batch_size = 32
        self.learning_rate = 0.001
        
        # Initialize Model
        self.model = TimeSeriesTransformer().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion_trend = nn.CrossEntropyLoss()
        self.criterion_vol = nn.MSELoss()

    def load_data(self, symbol="XAUUSD"):
        """
        Loads M1 candle data from the database for training.
        """
        if not os.path.exists(self.db_path):
            logger.error("Database not found!")
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            
            # Try loading from 'candles' table first (Historical Data)
            query = f"SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol='{symbol}' AND timeframe='M1' ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                # Fallback to 'ticks' table if candles are empty (Live Data only)
                logger.info("No candles found. Falling back to raw ticks...")
                query = f"SELECT timestamp, bid, ask FROM ticks WHERE symbol='{symbol}' ORDER BY timestamp ASC"
                df_ticks = pd.read_sql_query(query, conn)
                
                if df_ticks.empty:
                    logger.warning("No data found in database.")
                    conn.close()
                    return None
                    
                # Resample Ticks to Candles
                df_ticks['datetime'] = pd.to_datetime(df_ticks['timestamp'], unit='s')
                df_ticks.set_index('datetime', inplace=True)
                df_ticks['price'] = (df_ticks['bid'] + df_ticks['ask']) / 2
                
                df = df_ticks['price'].resample('1min').agg(['first', 'max', 'min', 'last'])
                df.columns = ['open', 'high', 'low', 'close']
                df['volume'] = 0 # Ticks don't have volume easily
                df.dropna(inplace=True)
            else:
                # Process Candle Data
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('datetime', inplace=True)
                
            conn.close()
            
            # Calculate Volatility (High - Low)
            df['volatility'] = df['high'] - df['low']
            
            return df
            
        except Exception as e:
            logger.error(f"Data Loading Error: {e}")
            return None

    def prepare_sequences(self, df):
        """
        Creates sliding window sequences for training.
        Input: 64 candles
        Target: Next candle's direction (0=Sell, 1=Neutral, 2=Buy) and Volatility
        
        CRITICAL FIX: Uses per-window normalization to match inference logic.
        """
        data = df[['open', 'high', 'low', 'close', 'volatility']].values
        
        sequences = []
        targets_trend = []
        targets_vol = []
        
        # Sliding Window
        for i in range(len(data) - self.seq_len - 1):
            # Extract raw window
            window = data[i : i + self.seq_len]
            
            # Per-Window Normalization (Z-Score)
            # This matches exactly what NexusBrain.predict() does
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0) + 1e-5 # Increased epsilon for stability
            norm_window = (window - mean) / std
            
            # Target is the NEXT candle (i + seq_len)
            next_candle = data[i + self.seq_len]
            current_close = window[-1, 3] # Last close in window
            
            # Trend Target
            # 0 = Sell (Next Close < Current Close - Threshold)
            # 1 = Neutral
            # 2 = Buy (Next Close > Current Close + Threshold)
            
            # Calculate threshold (e.g., 0.01% of price)
            threshold = current_close * 0.0001 
            
            if next_candle[3] > current_close + threshold:
                target = 2 # Buy
            elif next_candle[3] < current_close - threshold:
                target = 0 # Sell
            else:
                target = 1 # Neutral
                
            # Volatility Target (Next candle's volatility)
            # We normalize this too, relative to the window's volatility mean/std
            vol_mean = mean[4]
            vol_std = std[4]
            
            # Safety check for zero volatility
            if vol_std < 1e-6:
                target_vol = 0.0
            else:
                target_vol = (next_candle[4] - vol_mean) / vol_std
                
            # Clamp target volatility to prevent exploding gradients
            target_vol = np.clip(target_vol, -10.0, 10.0)
            
            sequences.append(norm_window)
            targets_trend.append(target)
            targets_vol.append(target_vol)
            
        return np.array(sequences), np.array(targets_trend), np.array(targets_vol)

    def train(self, epochs=10):
        logger.info("Starting Training Session...")
        df = self.load_data()
        if df is None or len(df) < 200:
            logger.warning("Not enough data to train yet. Collect more data.")
            return
            
        X, y_trend, y_vol = self.prepare_sequences(df)
        
        # Convert to Tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_trend_tensor = torch.LongTensor(y_trend).to(self.device)
        y_vol_tensor = torch.FloatTensor(y_vol).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_trend_tensor, y_vol_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        try:
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
                for batch_X, batch_trend, batch_vol in loader:
                    self.optimizer.zero_grad()
                    
                    pred_trend, pred_vol = self.model(batch_X)
                    
                    loss_trend = self.criterion_trend(pred_trend, batch_trend)
                    loss_vol = self.criterion_vol(pred_vol.squeeze(), batch_vol)
                    
                    # Weight the volatility loss lower to prioritize trend accuracy
                    loss = loss_trend + (0.1 * loss_vol)
                    
                    loss.backward()
                    
                    # Gradient Clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    if batch_count % 100 == 0:
                        logger.info(f"Epoch {epoch+1} | Batch {batch_count}/{len(loader)} | Current Loss: {loss.item():.4f}")
                    
                avg_loss = total_loss/len(loader)
                logger.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
                
            # Save Model
            torch.save(self.model.state_dict(), self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")
            
            return avg_loss
            
        except Exception as e:
            logger.critical(f"Training Crashed: {e}")
            import traceback
            logger.critical(traceback.format_exc())
            return float('inf')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    trainer = NexusTrainer()
    trainer.train()
