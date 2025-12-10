# AETHER Trading System - Version 4.0.4 (The Global Brain)

**Status**:  Production Ready  
**Version**: 4.0.4  
**Date**: December 02, 2025  

## Overview

This is **Version 4.0.4** of the AETHER AI Trading System, introducing "God Mode" intelligence and mathematical perfection in execution.

### New in v4.0.4
- **Layer 9 (Global Brain)**: Inter-Market Correlation Engine. Monitors DXY (Dollar) and US10Y (Yields) to predict Gold moves before they happen.
- **Velvet Cushion Protocol**: Dynamic slippage buffer that calculates the "Cost of Business" (Spread + Volatility) to ensure every close is net positive.
- **Iron Trap Geometry**: Fixed Hedge 3/4 logic to strictly enforce zone boundaries, eliminating "No Man's Land" exposure.
- **Council of Nine**: Replaced hard AI blocks with "Soft Penalties" (Weighted Voting) to prevent analysis paralysis while maintaining safety.
- **Unified Zone Recovery**: Aligned the "Trading Plan" logs perfectly with the Risk Manager's execution logic.

## Features

### Core Trading
- **Entry Logic**: AI Council (Nexus + Macro + Sentiment + Quantum)
- **Exit Logic**: Hybrid TP/SL (see BUCKET_TP_STRATEGY.md)
- **Risk Management**: Zone recovery, position sizing, max drawdown
- **Latency**: 50ms (broker TP) / 100ms (bucket monitoring)

### AI Components
- **NexusBrain**: Transformer-based prediction (663K params)
- **PPO Guardian**: Reinforcement learning for exits
- **Council**: Multi-agent consensus decision
- **Strategist**: Dynamic risk multipliers
- **IronShield**: Position sizing and recovery

## Quick Start

### 1. Install Dependencies
\\\ash
pip install -r requirements.txt
\\\

### 2. Configure
Edit \config/settings.yaml\:
- Set broker credentials (MT5)
- Configure risk parameters
- Set mode to PAPER or LIVE

### 3. Run Bot
\\\ash
python run_bot.py 3600
\\\

## Configuration

### Key Settings (\config/settings.yaml\)

\\\yaml
system:
  version: "4.0"
  mode: "PAPER"  # or "LIVE"

trading:
  symbol: "XAUUSD"
  timeframe: "M1"

risk:
  global_risk_percent: 2.0
  max_drawdown_percent: 5.0
  initial_lot: 0.01

ai_parameters:
  nexus_confidence_threshold: 0.30
  spread_max_points: 30
\\\

## Architecture

\\\
 run_bot.py              # Entry point (adaptive sleep: 5ms-1000ms)
 main.py                 # FastAPI server (optional)
 config/
    settings.yaml       # Main configuration
    secrets.env         # Credentials (gitignored)
 models/
    nexus_transformer.pth  # Trained model
    ppo_guardian.zip       # RL model
 data/                   # Runtime state (position_state.json)
 src/
     main_bot.py         # Bot orchestration (delegates to TradingEngine)
     trading_engine.py   # Trading logic (branching execution)
     position_manager.py # Position handling (async transitions)
     risk_manager.py     # Risk controls (dual-path logic)
     market_data.py      # Market data (ATR, trends)
     ai_core/           # AI modules (Council, Nexus, PPO)
     automation/        # Execution helpers
     bridge/            # Broker adapters (MT5, CCXT)
     infrastructure/    # Database (async queue), logging
\\\

## Trading Strategy

### Single Position Flow
1. Check existing positions  Decision branching:
   - IF positions exist: Manage existing (skip signal)
   - IF no positions: Continue to step 2
2. Council analyzes market  BUY/SELL signal
3. Execute with broker-side TP/SL
4. Broker watches 24/7 (50ms latency)
5. Hit TP  Instant execution
6. Python detects close  Log results

### Bucket Flow (Multiple Positions)
1. Initial position with broker TP
2. Price triggers zone recovery:
   - **Dual-path trigger calculation:**
     - Priority: Use stored triggers from hedge_plan
     - Fallback: Dynamic PPO calculation
3. **async transition_to_bucket():**
   - await broker.remove_tp_sl() (with rollback)
   - Execute hedge order
   - Transition SINGLE_ACTIVE  BUCKET_ACTIVE
4. Python monitoring takes over (5ms cycle)
5. Check break-even target (net profit  0.2 ATR)
6. Close all positions when target hit

See BUCKET_TP_STRATEGY.md for details.

## Risk Management

- **Max Drawdown**: 5% (configurable)
- **Position Limits**: Max 4 hedges per bucket
- **Spread Filter**: Max 3.0 pips
- **Cooldowns**: 15s between trades
- **Emergency Stop**: At 4+ positions

## Monitoring

### Logs
- ot_execution.log - Main bot activity
- [TRADE] - Entry/exit execution
- [BUCKET] - Multi-position management
- [TP HIT] - Profit targets
- [HEDGE] - Risk management

### Performance
Check session stats in logs:
- Win rate
- Average profit
- Total trades
- Sharpe ratio

## Production Checklist

- [ ] MT5 credentials configured
- [ ] Mode set to PAPER for testing
- [ ] Risk parameters validated
- [ ] Models present in \models/\
- [ ] Data directory writable
- [ ] Firewall allows MT5 connection
- [ ] Sufficient account balance
- [ ] Backtested on paper account

## Troubleshooting

### Bot Not Trading
- Check MT5 connection
- Verify spread < 3.0 pips
- Check confidence threshold (0.30)
- Ensure no active positions blocking

### TP Not Hitting
- Single position: Broker manages (check MT5)
- Bucket: Python monitors (check logs for exit conditions)

### High Drawdown
- Reduce \global_risk_percent\
- Increase \
exus_confidence_threshold\
- Check \max_layers\ in zone recovery

## Support

- Documentation: See \BUCKET_TP_STRATEGY.md\
- Issues: GitHub Issues
- Version: 4.0 (Stable)

## License

MIT License - Production Trading System
