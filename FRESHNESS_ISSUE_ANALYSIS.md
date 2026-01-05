# URGENT: Freshness Blocking Issue - Root Cause Analysis

**Date**: January 5, 2026, 7:55 AM IST  
**Issue**: Orders blocked due to stale tick data (43-52 seconds old)  
**Status**: ⚠️ **MARKET HOURS / DATA FEED ISSUE**

---

## 🔍 ROOT CAUSE

### **NOT a Threshold Problem**

The issue is **NOT** the freshness threshold. The real problems are:

1. **Tick Age**: 43-52 seconds (extremely stale)
2. **Time Offset**: -7199.07 seconds (almost 2 hours!)
3. **Pattern**: Continuously increasing age (43s → 52s)

### **Actual Causes**

#### **Cause #1: Market is Closed** (Most Likely)
- **Current Time**: 7:55 AM IST (Sunday)
- **Forex Market**: Closed on weekends
- **Gold (XAUUSD)**: Closed on weekends
- **Last Tick**: From Friday's close (~2 hours ago based on offset)

#### **Cause #2: MT5 Data Feed Issue**
- MT5 not receiving live ticks
- Broker server disconnected
- Network connectivity problem
- Symbol not subscribed

#### **Cause #3: Wrong Symbol/Timeframe**
- Symbol might not be actively traded
- Timeframe data not updating

---

## ✅ SOLUTIONS

### **Solution #1: Check Market Hours** (Recommended)

**Forex Market Hours** (GMT):
- **Opens**: Sunday 22:00 GMT (Monday 3:30 AM IST)
- **Closes**: Friday 22:00 GMT (Saturday 3:30 AM IST)

**Current Status**:
- **Today**: Sunday, January 5, 2026
- **Time**: 7:55 AM IST = 2:25 AM GMT
- **Market**: ❌ **CLOSED** (Opens in ~1 hour)

**Action**: Wait for market to open (Monday 3:30 AM IST)

---

### **Solution #2: Disable Freshness Gate for Testing**

If you need to test while market is closed:

**Option A: Environment Variable**
```bash
# Windows PowerShell
$env:AETHER_ENABLE_FRESHNESS_GATE="0"
python run_bot.py
```

**Option B: Create `.env` file**
```bash
# Create config/secrets.env or .env in root
AETHER_ENABLE_FRESHNESS_GATE=0
```

**Option C: Modify Code Temporarily**
```python
# In src/trading_engine.py line 165
self._freshness_gate = False  # Temporarily disable
```

---

### **Solution #3: Verify MT5 Connection**

**Check MT5 Terminal**:
1. Open MetaTrader 5
2. Check connection status (bottom right)
3. Verify symbol quotes are updating
4. Check "Quotes" window for live prices

**Test Connection**:
```python
import MetaTrader5 as mt5

# Initialize
mt5.initialize()

# Check connection
print(f"Connected: {mt5.terminal_info()}")

# Get last tick
tick = mt5.symbol_info_tick("XAUUSD")
print(f"Last tick: {tick}")
print(f"Time: {tick.time if tick else 'No data'}")

# Cleanup
mt5.shutdown()
```

---

### **Solution #4: Use Historical Data for Testing**

If you want to test logic while market is closed:

**Backtest Mode** (if implemented):
```bash
AETHER_BACKTEST_MODE=1
AETHER_ENABLE_FRESHNESS_GATE=0
python run_bot.py
```

---

## 🎯 RECOMMENDED ACTION

### **Immediate** (Right Now)

**Since market is closed on Sunday**:

1. **Stop the bot** (it can't trade anyway)
2. **Wait for market open**: Monday 3:30 AM IST
3. **Restart bot** when market opens

**OR**

1. **Disable freshness gate** for testing:
   ```bash
   $env:AETHER_ENABLE_FRESHNESS_GATE="0"
   python run_bot.py
   ```
2. **Re-enable** before live trading

---

### **When Market Opens** (Monday 3:30 AM IST)

1. **Verify MT5 is connected**
2. **Check live quotes are updating**
3. **Restart bot**
4. **Monitor for freshness issues**

If you still see blocking when market is OPEN:
- Tick age should be <5 seconds
- If still >5s, then MT5 connection issue
- Check broker server status

---

## 📊 Understanding the Logs

### **What the Logs Tell Us**

```
[FRESHNESS] NEW orders blocked (recovery/hedge): stale_tick age=43.14s max=5.00s (offset=-7199.07s)
```

**Breaking it down**:
- `age=43.14s` - Tick is 43 seconds old
- `max=5.00s` - Threshold is 5 seconds (correct)
- `offset=-7199.07s` - System time is 2 hours ahead of tick time

**Why offset is -7199s (2 hours)**:
- Last tick: Friday 22:00 GMT (market close)
- Current time: Sunday 02:25 GMT
- Difference: ~28 hours = market closed time

---

## 🔧 Quick Fixes

### **Fix #1: Disable Freshness Gate**
```powershell
# In PowerShell (Windows)
$env:AETHER_ENABLE_FRESHNESS_GATE="0"
python run_bot.py
```

### **Fix #2: Increase Threshold (Not Recommended)**
```powershell
# Allow up to 60 seconds (only for testing)
$env:AETHER_FRESH_TICK_MAX_AGE_S="60"
python run_bot.py
```

### **Fix #3: Wait for Market**
```bash
# Just wait until Monday 3:30 AM IST
# Market will open and ticks will be fresh
```

---

## ⚠️ IMPORTANT NOTES

### **DO NOT Trade When Market is Closed**
- Spreads are extremely wide
- No liquidity
- Orders may not execute
- High slippage risk

### **Freshness Gate is Working Correctly**
- It's **protecting you** from trading on stale data
- 43-52 second old ticks are **NOT safe** for trading
- The gate is doing its job

### **This is NOT a Bug**
- System is working as designed
- Market being closed is expected on weekends
- Freshness gate correctly blocks stale data

---

## 📅 Market Schedule

### **Forex Market Hours** (IST)
- **Opens**: Monday 3:30 AM IST
- **Closes**: Saturday 3:30 AM IST
- **Closed**: Saturday 3:30 AM - Monday 3:30 AM

### **Current Status** (Sunday 7:55 AM IST)
- **Market**: ❌ CLOSED
- **Opens in**: ~19.5 hours (Monday 3:30 AM)
- **Recommendation**: Wait for market open

---

## 🎯 FINAL RECOMMENDATION

### **For Testing** (Market Closed)
```bash
# Disable freshness gate
$env:AETHER_ENABLE_FRESHNESS_GATE="0"
python run_bot.py
```

### **For Live Trading** (Market Open)
```bash
# Keep freshness gate enabled (default)
# Just run normally
python run_bot.py
```

### **Best Practice**
1. **Test during market hours** with real data
2. **Keep freshness gate enabled** for safety
3. **Monitor tick age** - should be <2 seconds when market is active
4. **Stop bot** when market closes

---

**Summary**: Your bot is working correctly. The market is closed (Sunday). Wait for Monday 3:30 AM IST or disable freshness gate for testing only.
