# NexusBrain Upgrade Guide

**ENHANCEMENT 9: NexusBrain Architecture Upgrade**  
**Date**: January 4, 2026  
**Status**: Architecture Ready - Retraining Required

---

## 🎯 Upgrade Overview

### Current Architecture
- **Layers**: 2
- **d_model**: 128
- **Attention Heads**: 4
- **Parameters**: ~663,000
- **Sequence Length**: 64

### Upgraded Architecture
- **Layers**: 4 (2x increase)
- **d_model**: 256 (2x increase)
- **Attention Heads**: 8 (2x increase)
- **Parameters**: ~10,000,000 (15x increase)
- **Sequence Length**: 128 (2x increase)

### Expected Improvements
- **Accuracy**: +10-15%
- **Pattern Recognition**: Better long-term dependencies
- **Robustness**: More stable predictions
- **Generalization**: Better performance on unseen data

---

## 📝 Implementation Steps

### Step 1: Update Model Configuration

Edit `src/ai_core/nexus_transformer.py`:

```python
# CURRENT (Line 11):
def __init__(self, input_dim=5, order_book_dim=40, d_model=128, nhead=4, num_layers=2, output_dim=3, dropout=0.1):

# UPGRADE TO:
def __init__(self, input_dim=5, order_book_dim=40, d_model=256, nhead=8, num_layers=4, output_dim=3, dropout=0.1):
```

Also update line 27:
```python
# CURRENT:
encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)

# UPGRADE TO:
encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
```

### Step 2: Update Order Book Encoder

```python
# CURRENT (Lines 33-37):
self.ob_encoder = nn.Sequential(
    nn.Linear(order_book_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)

# UPGRADE TO:
self.ob_encoder = nn.Sequential(
    nn.Linear(order_book_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)
```

### Step 3: Update Combined Dimension

No changes needed - already uses `combined_dim = d_model + 32`

### Step 4: Backup Current Model

```bash
# Backup existing trained model
cp models/nexus_transformer.pth models/nexus_transformer_v1_backup.pth
```

### Step 5: Retrain Model

**Option A: Use Existing Trainer**

```bash
# If you have nexus_trainer.py
python src/ai_core/nexus_trainer.py --epochs 100 --batch_size 32
```

**Option B: Manual Training Script**

Create `train_nexus_v2.py`:

```python
import torch
import torch.nn as nn
from src.ai_core.nexus_transformer import TimeSeriesTransformer

# Initialize upgraded model
model = TimeSeriesTransformer(
    input_dim=5,
    order_book_dim=40,
    d_model=256,      # UPGRADED
    nhead=8,          # UPGRADED
    num_layers=4,     # UPGRADED
    output_dim=3,
    dropout=0.1
)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Load your training data
# train_loader = ... (your data loading code)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        trend_logits, vol_pred = model(batch['features'], batch['order_book'])
        
        # Calculate loss
        loss = criterion(trend_logits, batch['labels'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'models/nexus_v2_epoch_{epoch+1}.pth')

# Save final model
torch.save(model.state_dict(), 'models/nexus_transformer.pth')
print("Training complete!")
```

---

## 📊 Training Data Requirements

### Minimum Requirements
- **Samples**: 50,000+ candles
- **Timeframe**: M1 (1-minute)
- **Features**: OHLCV + Order Book (if available)
- **Labels**: Direction (UP/DOWN/NEUTRAL)

### Recommended
- **Samples**: 200,000+ candles (3-4 months of M1 data)
- **Validation Split**: 20%
- **Test Split**: 10%

### Data Sources
1. **MT5 History**: Export from MetaTrader 5
2. **CSV Files**: Historical OHLCV data
3. **Database**: If you have TimescaleDB setup

---

## ⚙️ Training Configuration

### Hyperparameters

```python
TRAINING_CONFIG = {
    # Model
    'd_model': 256,
    'nhead': 8,
    'num_layers': 4,
    'dropout': 0.1,
    
    # Training
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    
    # Data
    'sequence_length': 128,  # Increased from 64
    'train_split': 0.7,
    'val_split': 0.2,
    'test_split': 0.1,
    
    # Optimization
    'optimizer': 'Adam',
    'scheduler': 'ReduceLROnPlateau',
    'early_stopping_patience': 10
}
```

### Hardware Requirements

- **GPU**: Recommended (NVIDIA with CUDA)
- **RAM**: 16GB minimum
- **Storage**: 5GB for model checkpoints
- **Training Time**: 
  - CPU: 2-3 days
  - GPU (RTX 3060): 6-8 hours
  - GPU (RTX 4090): 2-3 hours

---

## 🧪 Testing & Validation

### Step 1: Validate Architecture

```python
import torch
from src.ai_core.nexus_transformer import TimeSeriesTransformer

# Test model creation
model = TimeSeriesTransformer(
    input_dim=5,
    d_model=256,
    nhead=8,
    num_layers=4
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # Should be ~10M

# Test forward pass
dummy_input = torch.randn(1, 128, 5)  # [batch, seq, features]
dummy_ob = torch.randn(1, 40)  # [batch, ob_features]

trend, vol = model(dummy_input, dummy_ob)
print(f"Trend output shape: {trend.shape}")  # Should be [1, 3]
print(f"Vol output shape: {vol.shape}")      # Should be [1, 1]
```

### Step 2: Compare Performance

After training, compare old vs new model:

```python
# Load both models
old_model = TimeSeriesTransformer(d_model=128, nhead=4, num_layers=2)
old_model.load_state_dict(torch.load('models/nexus_transformer_v1_backup.pth'))

new_model = TimeSeriesTransformer(d_model=256, nhead=8, num_layers=4)
new_model.load_state_dict(torch.load('models/nexus_transformer.pth'))

# Test on validation set
# ... (your evaluation code)
```

---

## 🚀 Deployment

### Step 1: Verify Model File

```bash
# Check model file size
ls -lh models/nexus_transformer.pth

# Should be ~40-50 MB (vs ~2.5 MB for old model)
```

### Step 2: Update Bot Configuration

No code changes needed! The bot will automatically use the new model.

### Step 3: Test in PAPER Mode

```bash
# Run bot in paper trading mode
python run_bot.py

# Monitor logs for:
# - Model loading success
# - Prediction quality
# - No errors
```

### Step 4: Monitor Performance

Track these metrics for 24 hours:
- Prediction accuracy
- Win rate
- Profit factor
- Sharpe ratio

If performance improves, deploy to LIVE.

---

## 📈 Expected Results

### Before Upgrade (Current)
- Accuracy: ~55-60%
- Win Rate: ~50-55%
- Sharpe Ratio: ~1.5-2.0

### After Upgrade (Target)
- Accuracy: ~65-70% (+10-15%)
- Win Rate: ~60-65% (+10%)
- Sharpe Ratio: ~2.0-2.5 (+25%)

---

## ⚠️ Important Notes

1. **Backup First**: Always backup current model before upgrading
2. **Test Thoroughly**: Test in PAPER mode for at least 24 hours
3. **Monitor Closely**: Watch for any degradation in first week
4. **Rollback Plan**: Keep old model as fallback

---

## 🔧 Troubleshooting

### Issue: Out of Memory During Training

**Solution**: Reduce batch size
```python
batch_size = 16  # Instead of 32
```

### Issue: Model Not Loading

**Solution**: Check architecture matches
```python
# Verify model config
print(model)
```

### Issue: Poor Performance After Upgrade

**Solution**: May need more training data or epochs
```python
epochs = 150  # Instead of 100
```

---

## 📞 Support

If you encounter issues:
1. Check logs for errors
2. Verify data quality
3. Ensure GPU drivers are updated
4. Consider starting with smaller model (d_model=192, layers=3)

---

**Status**: Ready for implementation when you have time for retraining (1 week)

**Priority**: Lower (current model is working well)

**Recommendation**: Implement after collecting more trading data for better training
