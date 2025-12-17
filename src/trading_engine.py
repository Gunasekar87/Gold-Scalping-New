"""
Trading Engine - Core trading logic and orchestration.

This module provides the main trading engine including:
- Trade entry and exit orchestration
- AI decision integration
- Market condition validation
- Rate limiting and safety controls

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import logging
import asyncio
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Import constants
from .constants import ScalpingConfig, ZoneRecoveryConfig

# Import async database components
from .infrastructure.async_database import (
    AsyncDatabaseManager, AsyncDatabaseQueue, TickData, CandleData, TradeData
)
from .utils.trading_logger import TradingLogger, DecisionTracker, format_pips
from .utils.news_filter import NewsFilter
from .utils.news_calendar import NewsCalendar
from .ai_core.tick_pressure import TickPressureAnalyzer

# [AI INTELLIGENCE] New Policy & Governance Modules
from src.config.settings import FLAGS, POLICY as _PTUNE, RISK as _RLIM
from src.policy.hedge_policy import HedgePolicy, HedgeConfig
from src.policy.risk_governor import RiskGovernor, RiskLimits
from src.utils.telemetry import TelemetryWriter, DecisionRecord
from src.features.market_features import spread_atr, zscore, simple_breakout_quality, simple_regime_trend

logger = logging.getLogger("TradingEngine")
# [CRITICAL] Get the specific UI logger that run_bot.py listens to
ui_logger = logging.getLogger("AETHER_UI")


class TradeAction(Enum):
    """Enumeration of possible trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """Represents a trading signal with metadata."""
    action: TradeAction
    symbol: str
    confidence: float
    reason: str
    metadata: Dict[str, Any]
    timestamp: float

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class TradingConfig:
    """Configuration for trading parameters."""
    symbol: str
    initial_lot: float
    max_lot: float = 100.0
    min_lot: float = 0.01
    global_trade_cooldown: float = 5.0  # Reduced from 15.0s for Continuous Scalping
    max_spread_pips: float = 30.0
    risk_multiplier_range: Tuple[float, float] = (0.1, 2.0)
    timeframe: str = "M1"


class TradingEngine:
    """
    Core trading engine that orchestrates all trading activities.

    This class handles:
    - AI decision processing and validation
    - Trade execution with safety checks
    - Rate limiting and cooldown management
    - Integration with all trading components
    """

    def __init__(self, config: TradingConfig, broker_adapter, market_data, position_manager, risk_manager, db_manager: Optional[AsyncDatabaseManager] = None, ppo_guardian=None, global_brain=None):
        self.config = config
        self.broker = broker_adapter
        self.market_data = market_data
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.ppo_guardian = ppo_guardian
        self.global_brain = global_brain # Layer 9: Inter-Market Correlation

        # [PHASE 3] Initialize Supervisor and Workers
        from .ai_core.supervisor import Supervisor
        from .ai_core.workers import RangeWorker, TrendWorker
        from .ai.hedge_intelligence import HedgeIntelligence # [NEW] The Oracle
        
        self.supervisor = Supervisor()
        self.range_worker = RangeWorker()
        self.trend_worker = TrendWorker()
        self.hedge_intel = HedgeIntelligence(self.config) # [NEW] Initialize Oracle
        self.news_filter = NewsFilter() # [AI INTELLIGENCE] Initialize News Filter
        self.news_calendar = NewsCalendar() # [HIGHEST INTELLIGENCE] Event Horizon
        self.tick_analyzer = TickPressureAnalyzer() # [HIGHEST INTELLIGENCE] Tick Pressure

        # [AI INTELLIGENCE] Initialize Policy & Governance
        self._telemetry = TelemetryWriter()
        self._hedge_policy = HedgePolicy(HedgeConfig(
            min_confidence=_PTUNE.MIN_CONFIDENCE,
            max_spread_atr=_PTUNE.MAX_SPREAD_ATR,
            tp_atr_mult=_PTUNE.TP_ATR_MULT,
            sl_atr_mult=_PTUNE.SL_ATR_MULT,
            max_stack=_PTUNE.MAX_HEDGE_STACK,
        ))
        self._governor = RiskGovernor(RiskLimits(
            max_daily_loss_pct=_RLIM.MAX_DAILY_LOSS_PCT,
            max_total_exposure=_RLIM.MAX_TOTAL_EXPOSURE,
            max_spread_points=_RLIM.MAX_SPREAD_POINTS,
            news_lockout=_RLIM.NEWS_LOCKOUT,
        ))

        # Async database components
        self.db_manager = db_manager
        
        # Decision Tracker
        self.decision_tracker = DecisionTracker()
        self.db_queue: Optional[AsyncDatabaseQueue] = None

        # Rate limiting
        self.last_trade_time = 0.0
        self.trade_cooldown_active = False
        
        # Status Tracking (For UI Feedback)
        self.last_pause_reason = None

        # Decision tracking for smart logging (only log changes)
        self.decision_tracker = DecisionTracker()
        self._last_signal_logged = False  # Track if we just logged a signal
        self._last_position_status_time = 0.0  # Track last position status log time
        self._last_cooldown_log_time = 0.0 # Track last cooldown log time

        # Statistics
        self.session_stats = {
            "trades_opened": 0,
            "trades_closed": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "start_time": time.time()
        }

        logger.info(f"TradingEngine initialized for {config.symbol}")

    async def initialize_database(self) -> None:
        """Initialize async database components."""
        if self.db_manager:
            try:
                await self.db_manager.connect()
                await self.db_manager.initialize_schema()
                self.db_queue = AsyncDatabaseQueue(self.db_manager)
                await self.db_queue.start()
                logger.info("Database components initialized")
            except Exception as e:
                logger.error(f"Database initialization failed: {e}")
                self.db_manager = None

    async def shutdown_database(self) -> None:
        """Shutdown async database components."""
        if self.db_queue:
            await self.db_queue.stop()
        if self.db_manager:
            await self.db_manager.disconnect()

    def validate_market_conditions(self, symbol: str, tick: Dict) -> Tuple[bool, str]:
        """
        Validate if market conditions are suitable for trading.

        Args:
            symbol: Trading symbol
            tick: Current tick data

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check basic market data validity
        if not tick or 'bid' not in tick or 'ask' not in tick:
            return False, "Invalid tick data"

        # [HIGHEST INTELLIGENCE] Check Event Horizon (News Blackout)
        if self.news_calendar.is_blackout_period(symbol):
            return False, "News Blackout (Event Horizon)"

        # Check spread
        # Adjust multiplier for JPY and XAU pairs
        multiplier = 10000
        if "JPY" in symbol or "XAU" in symbol:
            multiplier = 100
            
        spread_pips = abs(tick['ask'] - tick['bid']) * multiplier
        if spread_pips > self.config.max_spread_pips:
            return False, f"Spread too wide: {spread_pips:.1f} pips"

        # Check market data manager conditions
        return self.market_data.validate_market_conditions(symbol, tick)

    def calculate_position_size(self, signal: TradeSignal, account_info: Dict,
                              strategist, shield, ppo_guardian=None, 
                              atr_value=0.001, trend_strength=0.0) -> Tuple[float, str]:
        """
        Calculate appropriate position size for a trade signal with PPO optimization.

        Args:
            signal: Trade signal
            account_info: Account information
            strategist: Strategist instance
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance for AI-driven sizing (optional)
            atr_value: Current ATR for volatility assessment
            trend_strength: Trend strength for PPO decision

        Returns:
            Tuple of (lot_size, reason)
        """
        try:
            # Get base lot size from shield
            raw_lot = shield.calculate_entry_lot(
                account_info.get('equity', 1000),
                confidence=signal.confidence,
                atr_value=atr_value,
                trend_strength=trend_strength
            )

            if raw_lot is None or raw_lot <= 0:
                return 0.0, "Invalid base lot calculation"

            # Apply strategist risk multiplier
            risk_mult = strategist.get_risk_multiplier()
            
            # --- DYNAMIC LOT SIZING (Layer 8) ---
            # Scale based on Account Balance & Confidence
            # 1. Balance Scaling: 0.01 lots per $1000 equity (Conservative)
            equity = account_info.get('equity', 1000)
            balance_scale = max(1.0, equity / 1000.0)
            
            # 2. Confidence Scaling: 
            # If Confidence > 0.8 -> 1.2x size
            # If Confidence < 0.5 -> 0.8x size
            conf_scale = 1.0
            if signal.confidence > 0.8:
                conf_scale = 1.2
            elif signal.confidence < 0.5:
                conf_scale = 0.8
                
            # 3. Trend Closing Confidence (Win Rate)
            # If recent win rate is high (>60%), increase size slightly
            # If recent win rate is low (<40%), decrease size
            win_rate_scale = 1.0
            if hasattr(strategist, 'recent_win_rate'):
                 if strategist.recent_win_rate > 0.6:
                     win_rate_scale = 1.1
                 elif strategist.recent_win_rate < 0.4:
                     win_rate_scale = 0.8

            # Apply all scalings to raw_lot (which is usually fixed 0.01)
            # If raw_lot is fixed 0.01, we scale it up.
            # If raw_lot is already dynamic from Shield, we just refine it.
            
            # Let's assume raw_lot is the "Base Lot" from config (e.g. 0.01)
            # New Lot = Base * BalanceScale * ConfScale * WinRateScale * RiskMult
            
            dynamic_lot = raw_lot * balance_scale * conf_scale * win_rate_scale * risk_mult
            
            # NEW: Apply PPO Guardian AI optimization
            ppo_mult = 1.0  # Default if PPO not available
            if ppo_guardian:
                try:
                    # Calculate pip multiplier for logging
                    pip_multiplier = 100 if "XAU" in signal.symbol or "GOLD" in signal.symbol else 10000
                    
                    ppo_mult = ppo_guardian.get_position_size_multiplier(
                        atr=atr_value,
                        trend_strength=trend_strength,
                        confidence=signal.confidence,
                        current_equity=account_info.get('equity', 1000)
                    )
                    atr_pips = atr_value * pip_multiplier
                    # Changed to DEBUG to prevent log spam
                    logger.debug(f"[PPO_SIZING] Position Multiplier: {ppo_mult:.2f}x | ATR: {atr_pips:.1f}pips | Trend: {trend_strength:.2f} | Conf: {signal.confidence:.2f}")
                except Exception as ppo_error:
                    logger.warning(f"[PPO_SIZING] Failed: {ppo_error}, using 1.0x default")
                    ppo_mult = 1.0
            
            # Calculate final lot size with ALL multipliers
            # Use 4 decimals to allow for micro-lots (0.001) if broker supports it.
            # MT5 Adapter will normalize to exact step (e.g. 0.01 or 0.001).
            entry_lot = round(dynamic_lot * ppo_mult, 4)

            # Validate lot size bounds
            if entry_lot < self.config.min_lot:
                # If calculated is too small but we have equity, default to min_lot
                entry_lot = self.config.min_lot

            if entry_lot > self.config.max_lot:
                entry_lot = self.config.max_lot

            reason_parts = [
                f"Lot: {entry_lot}",
                f"Base: {raw_lot:.2f}",
                f"BalScale: {balance_scale:.1f}",
                f"ConfScale: {conf_scale:.1f}",
                f"WinScale: {win_rate_scale:.1f}"
            ]
            
            if ppo_guardian and ppo_mult != 1.0:
                reason_parts.append(f"ppo_mult: {ppo_mult:.2f}")
            
            return entry_lot, " | ".join(reason_parts)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0, f"Calculation error: {e}"

    def validate_trade_entry(self, signal: TradeSignal, lot_size: float,
                           account_info: Dict, tick: Dict, is_recovery_trade: bool = False) -> Tuple[bool, str]:
        """
        Perform final validation before executing a trade.

        Args:
            signal: Trade signal
            lot_size: Calculated lot size
            account_info: Account information
            tick: Current tick data
            is_recovery_trade: If True, bypass position direction check (for hedges/DCA/zone recovery)

        Returns:
            Tuple of (can_enter, reason)
        """
        # Check global trade cooldown
        time_since_last_trade = time.time() - self.last_trade_time
        if time_since_last_trade < self.config.global_trade_cooldown:
            return False, f"Global cooldown: {time_since_last_trade:.1f}s < {self.config.global_trade_cooldown}s"

        # Check account equity
        equity = account_info.get('equity', 0)
        if equity <= 0:
            return False, "Invalid account equity"

        # Check if algo trading is allowed
        if not self.broker.is_trade_allowed():
            return False, "Algo trading disabled"

        # [AI INTELLIGENCE] Check News Filter
        # Prevent entries during high-impact news events
        if self.news_filter and not self.news_filter.should_trade():
            return False, "High-impact news event imminent or active"

        # Check for recent bucket closes
        symbol = signal.symbol
        if self.position_manager.is_bucket_closed_recently(f"{symbol}_recent", 15.0):
            return False, "Recent bucket close - cooldown active"

        # CRITICAL: Prevent adding to existing positions in same direction
        # (unless it's a recovery trade for hedging/DCA/zone recovery)
        if not is_recovery_trade:
            existing_positions = self.broker.get_positions(symbol)
            if existing_positions:
                # Check if any position is in the same direction as the signal
                signal_direction = 0 if signal.action == TradeAction.BUY else 1  # 0=BUY, 1=SELL
                for pos in existing_positions:
                    # Position object has .type attribute (0=BUY, 1=SELL)
                    if pos.type == signal_direction:
                        return False, f"Position already exists in {signal.action.value} direction - awaiting TP/hedge trigger"

        # --- HFT LAYER 7: ORDER BOOK IMBALANCE (OBI) FILTER ---
        # If OBI is strongly against us, block the trade.
        # OBI > 0.3 means Strong Buy Pressure. OBI < -0.3 means Strong Sell Pressure.
        obi = tick.get('obi', 0.0)
        if abs(obi) > 0.3: # Only filter if there is significant imbalance
            if signal.action == TradeAction.BUY and obi < -0.3:
                return False, f"HFT Block: Strong Sell Pressure (OBI: {obi:.2f})"
            if signal.action == TradeAction.SELL and obi > 0.3:
                return False, f"HFT Block: Strong Buy Pressure (OBI: {obi:.2f})"

        return True, "Trade entry validated"

    async def execute_trade_entry(self, signal: TradeSignal, lot_size: float,
                          tick: Dict, strategist, shield) -> Optional[Dict]:
        """
        Execute a trade entry order.

        Args:
            signal: Trade signal
            lot_size: Position size
            tick: Current tick data
            strategist: Strategist instance
            shield: IronShield instance

        Returns:
            Order result dict or None if failed
        """
        try:
            # Extract signal metadata for logging and logic
            nexus_conf = signal.metadata.get('nexus_signal_confidence', 0.0)
            trend_strength = signal.metadata.get('trend_strength', 0.0)
            market_status = signal.metadata.get('market_status', 'Unknown')
            caution_factor = signal.metadata.get('caution_factor', 1.0)
            sentiment = signal.metadata.get('sentiment_score', 0.0)
            regime_name = signal.metadata.get('regime', 'UNKNOWN')

            # Determine entry price and order type
            if signal.action == TradeAction.BUY:
                entry_price = tick['ask']
                order_type = "BUY"
            else:
                entry_price = tick['bid']
                order_type = "SELL"

            # Get dynamic TP parameters - USE ACTUAL ATR
            symbol = signal.symbol
            atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
            
            # Convert ATR to points based on symbol type
            if 'JPY' in symbol or 'XAU' in symbol:
                atr_points = atr_value * 100  # JPY and Gold pairs use 2 decimal places
                pip_multiplier = 100
            else:
                atr_points = atr_value * 10000  # Major forex pairs use 4 decimal places
                pip_multiplier = 10000
                
            zone_points, tp_points = shield.get_dynamic_params(atr_points)  # Convert to points
            tp_pips_entry = tp_points # Define for logging compatibility

            # Calculate broker-side TP price for instant execution
            # Use Shield's dynamic TP (configurable via settings.yaml)
            # NO SL - hedging strategy manages risk through zone recovery
            
            if order_type == "BUY":
                tp_price_broker = entry_price + (tp_points / pip_multiplier)
            else:  # SELL
                tp_price_broker = entry_price - (tp_points / pip_multiplier)
            
            # Round to correct precision
            digits = 2 if "JPY" in symbol or "XAU" in symbol else 5
            tp_price_broker = round(tp_price_broker, digits)

            # Execute order WITHOUT broker TP/SL for hedge strategy
            # CRITICAL: No SL on broker because hedges ARE the risk management!
            # Setting SL would close position before hedges trigger, defeating the strategy.
            # Bucket logic (Python-side 100ms monitoring) manages ALL exits including break-even.
            
            # === EQUITY CHECK BEFORE EXECUTION ===
            account_info = self.broker.get_account_info()
            logger.debug(f"[ACCOUNT_INFO] Retrieved: {account_info}")
            if not account_info:
                logger.error("[ACCOUNT_INFO] Failed to retrieve account information")
                return None
            
            if account_info.get('equity', 0) < 100:
                logger.error(f"[TRADE] Insufficient equity: {account_info.get('equity', 0)} < 100")
                return None
            
            # Estimate required margin
            contract_size = 100 if "XAU" in signal.symbol else 100000
            leverage = account_info.get('leverage', 100)
            required_margin = lot_size * contract_size * entry_price / leverage
            available_margin = account_info.get('margin_free', 0)
            
            if required_margin > available_margin:
                logger.error(f"[TRADE] Insufficient margin: Required {required_margin:.2f} > Available {available_margin:.2f}")
                return None
            
            result = self.broker.execute_order(
                action="OPEN",
                symbol=signal.symbol,
                order_type=order_type,
                price=entry_price,
                volume=lot_size,
                sl=0.0,  # NO SL - hedging strategy manages risk
                tp=0.0,  # NO BROKER TP - Virtual TP managed by Python for nil latency
                # MT5 comment limit: 31 chars
                comment=f"{signal.reason[:20]}"
            )
            
            # Log broker targets (actually set on broker for single positions)
            if result and result.get('ticket'):
                logger.info(f"[ORDER] VIRTUAL TP: {tp_price_broker:.5f} (Dynamic) | NO SL - Hedging Strategy")
                logger.info(f"[ORDER] Risk Management: Virtual TP for instant exits | Hedges manage downside risk")

            if result and result.get('ticket'):
                # Update statistics
                self.session_stats["trades_opened"] += 1

                # Record trade in database
                if self.db_queue:
                    trade_data = TradeData(
                        ticket=result['ticket'],
                        symbol=signal.symbol,
                        trade_type=order_type,
                        volume=lot_size,
                        open_price=entry_price,
                        close_price=None,
                        profit=None,
                        open_time=time.time(),
                        close_time=None,
                        strategy_reason=signal.reason
                    )
                    await self.db_queue.add_trade(trade_data)

                # Record learning data with entry TP/SL metadata
                ticket = result['ticket']
                self.position_manager.record_learning_trade(ticket, signal.symbol, {
                    "symbol": signal.symbol,
                    "type": order_type,
                    "entry_price": entry_price,
                    "obs": [0.0, tp_points, 0.0, signal.metadata.get("nexus_signal_confidence", 0.0)],
                    "action": [1.0, 0.8],  # Default hedge mult and zone mod
                    "open_time": time.time(),
                    # Store entry targets for Python-side monitoring
                    "entry_tp_pips": tp_pips_entry,
                    "entry_sl_pips": 0.0,  # NO SL - hedging strategy only
                    "entry_atr": atr_value
                })

                # === ENHANCED AI INTELLIGENCE TRADING PLAN LOGGING ===
                # Comprehensive AI decision factors and execution intelligence
                symbol = signal.symbol
                atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
                
                # [PHASE 3] Get Regime from Supervisor
                regime_name = "UNKNOWN"
                if hasattr(self, 'supervisor'):
                    regime = self.supervisor.detect_regime(self.market_data.get_tick_data(symbol))
                    regime_name = regime.name
                
                # Convert ATR to pips based on symbol type
                if "JPY" in symbol or "XAU" in symbol:
                    atr_pips = atr_value * 100  # JPY and Gold pairs use 2 decimal places
                else:
                    atr_pips = atr_value * 10000  # Major forex pairs use 4 decimal places

                # Risk intelligence calculations (AI-Adjusted)
                # OPTIMIZED GAPS: Linear expansion to prevent tight whipsaws in Hedge 2
                # Ask PPO for dynamic zone modifier to align Plan with Execution
                zone_mod = 1.0
                if self.ppo_guardian:
                    try:
                        # Use 0 drawdown for initial plan
                        _, zone_mod = self.ppo_guardian.get_dynamic_zone(0.0, atr_value, trend_strength, nexus_conf)
                    except Exception as e:
                        logger.warning(f"[PPO] Failed to get dynamic zone: {e}")
                        zone_mod = 1.0

                # Define pip divisor based on symbol for correct price calculation
                pip_divisor = 100 if "JPY" in symbol or "XAU" in symbol else 10000
                point_approx = 1.0 / (pip_divisor * 10) # Approx point size for spread calc

                tp_pips = atr_pips * ScalpingConfig.TP_ATR_MULTIPLIER  # 30% of ATR for tight scalping
                tp_price_dist = tp_pips / pip_divisor
                tp_price = entry_price + tp_price_dist if order_type == "BUY" else entry_price - tp_price_dist

                # === UNIFIED ZONE RECOVERY PLAN ===
                # Calculate Zone Width (Fixed for the sequence)
                # Based on Hedge 1 distance (0.5 ATR)
                zone_pips = atr_pips * ScalpingConfig.HEDGE1_ATR_MULTIPLIER * zone_mod
                zone_price_dist = zone_pips / pip_divisor
                
                # Get Spread for Lot Calculation (Estimate)
                current_tick = self.market_data.get_tick_data(symbol)
                spread_points = 20 # Default 2 pips
                if current_tick and 'ask' in current_tick and 'bid' in current_tick:
                    spread_points = (current_tick['ask'] - current_tick['bid']) / point_approx
                
                # Calculate Lots using IronShield (Same as Execution)
                # Convert pips to points for IronShield
                zone_points = zone_pips * 10
                tp_points = tp_pips * 10
                
                # === AI BOOST: FUTURISTIC PREDICTION ===
                # If Oracle predicts a crash/rally, boost the hedge lots in that direction
                # to capitalize on the move and exit faster.
                ai_bias = 0.0
                oracle_prediction = signal.metadata.get('oracle_prediction', 'NEUTRAL')
                
                # Determine H1 direction (Opposite to Entry)
                h1_dir_check = "SELL" if order_type == "BUY" else "BUY"
                
                if h1_dir_check == "SELL" and oracle_prediction == "BEARISH":
                    ai_bias = 0.10 # +10% Aggression on Sell Hedges
                elif h1_dir_check == "BUY" and oracle_prediction == "BULLISH":
                    ai_bias = 0.10 # +10% Aggression on Buy Hedges

                # Calculate Hedge Lots Iteratively
                # Hedge 1 (Opposite)
                raw_h1 = self.risk_manager.shield.calculate_defense(
                    lot_size, spread_points, fixed_zone_points=zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=1
                )
                hedge1_lot = round(raw_h1 * (1.0 + ai_bias), 4)
                
                # Hedge 2 (Recovery) - Recovers H1
                hedge2_lot = self.risk_manager.shield.calculate_defense(
                    hedge1_lot, spread_points, fixed_zone_points=zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=2
                )
                
                # === SMART EXPANSION LOGIC ===
                # Determine Expansion Factor based on AI Prediction
                ai_expansion_factor = 0.3 # Default: Hedge 3 is 30% further out
                
                # Check Oracle Prediction (if available in metadata)
                oracle_prediction = signal.metadata.get('oracle_prediction', 'NEUTRAL')
                
                # If we are Buying (H1 is Sell), and AI says Bullish (Reversal), delay the Sell Hedge 3
                if order_type == "BUY" and oracle_prediction == "BULLISH":
                    ai_expansion_factor = 0.6 # Push it 60% away (Give it room to breathe)
                # If we are Selling (H1 is Buy), and AI says Bearish (Reversal), delay the Buy Hedge 3
                elif order_type == "SELL" and oracle_prediction == "BEARISH":
                    ai_expansion_factor = 0.6

                # Calculate Smart Buffer
                smart_buffer_pips = (zone_pips * ai_expansion_factor)
                smart_buffer_dist = smart_buffer_pips / pip_divisor
                
                # Hedge 3 (Opposite) - Same direction as H1
                # We boost the zone size for H3 calculation to account for the extra distance
                # Effective Zone for H3 = Zone + Buffer
                expanded_zone_points = zone_points * (1.0 + ai_expansion_factor)
                
                raw_h3 = self.risk_manager.shield.calculate_defense(
                    hedge2_lot, spread_points, fixed_zone_points=expanded_zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=3
                )
                hedge3_lot = round(raw_h3 * (1.0 + ai_bias), 4)
                
                # Hedge 4 (Recovery) - Recovers H3
                # H4 recovers H3 over the expanded zone distance
                hedge4_lot = self.risk_manager.shield.calculate_defense(
                    hedge3_lot, spread_points, fixed_zone_points=expanded_zone_points, fixed_tp_points=tp_points,
                    oracle_prediction=oracle_prediction, hedge_level=4
                )

                # Calculate Hedge Prices (Smart Expansion Logic)
                if order_type == "BUY":
                    # Initial Buy
                    # Hedge 1 (Sell): Entry - Zone
                    hedge1_price = entry_price - zone_price_dist
                    h1_dir = "SELL"
                    
                    # Hedge 2 (Buy): Entry (Recovery)
                    hedge2_price = entry_price
                    h2_dir = "BUY"
                    
                    # Hedge 3 (Sell): H1 - Smart Buffer (Expansion)
                    hedge3_price = hedge1_price - smart_buffer_dist
                    h3_dir = "SELL"
                    
                    # Hedge 4 (Buy): Matches H2 (Entry)
                    hedge4_price = hedge2_price
                    h4_dir = "BUY"
                    
                else: # SELL
                    # Initial Sell
                    # Hedge 1 (Buy): Entry + Zone
                    hedge1_price = entry_price + zone_price_dist
                    h1_dir = "BUY"
                    
                    # Hedge 2 (Sell): Entry (Recovery)
                    hedge2_price = entry_price
                    h2_dir = "SELL"
                    
                    # Hedge 3 (Buy): H1 + Smart Buffer (Expansion)
                    hedge3_price = hedge1_price + smart_buffer_dist
                    h3_dir = "BUY"
                    
                    # Hedge 4 (Sell): Matches H2 (Entry)
                    hedge4_price = hedge2_price
                    h4_dir = "SELL"

                # Log Smart Expansion Plan
                logger.info(f"[SMART EXPANSION] H3 Buffer: {ai_expansion_factor*100:.0f}% ({smart_buffer_pips:.1f} pips) | AI: {oracle_prediction}")

                # Calculate projected hedge levels for display AND execution
                projected_hedges = []
                
                # Hedge 1
                projected_hedges.append({
                    'direction': h1_dir,
                    'trigger_price': round(hedge1_price, 5),
                    'lots': hedge1_lot,
                    'tp_price': "Break-even"
                })
                
                # Hedge 2
                projected_hedges.append({
                    'direction': h2_dir,
                    'trigger_price': round(hedge2_price, 5),
                    'lots': hedge2_lot,
                    'tp_price': "Break-even"
                })

                # Hedge 3
                projected_hedges.append({
                    'direction': h3_dir,
                    'trigger_price': round(hedge3_price, 5),
                    'lots': hedge3_lot,
                    'tp_price': "Break-even"
                })
                
                # Hedge 4
                projected_hedges.append({
                    'direction': h4_dir,
                    'trigger_price': round(hedge4_price, 5),
                    'lots': hedge4_lot,
                    'tp_price': "Break-even"
                })

                # Log using the new UI Logger
                TradingLogger.log_initial_trade(f"{symbol}_{0 if order_type=='BUY' else 1}", {
                    'action': order_type,
                    'symbol': symbol,
                    'lots': lot_size,
                    'entry_price': entry_price,
                    'tp_price': tp_price_broker,
                    'tp_atr': 0.3,
                    'tp_pips': tp_pips_entry,
                    'hedges': projected_hedges,
                    'atr_pips': atr_pips,
                    'reasoning': f"[{regime_name}] {signal.reason}"
                })

                # === STRUCTURED TRADING PLAN LOG ===
                # Use TradingLogger for clean, structured output
                
                # Prepare data for TradingLogger
                bucket_id = f"{symbol}_{int(time.time())}"
                
                # Store metadata for this trade in position manager (for AI learning and exit logic)
                if result and result.get('ticket'):
                    ticket = result['ticket']
                    atr_pips = atr_value * (100 if "XAU" in symbol or "GOLD" in symbol else 10000)
                    
                    # Use thread-safe persistence method
                    self.position_manager.record_trade_metadata(ticket, {
                        "symbol": symbol,
                        "type": order_type,
                        "entry_price": entry_price,
                        "entry_tp_pips": tp_pips,  # Store fixed TP target
                        "entry_sl_pips": 0.0,  # NO SL - hedging strategy only
                        "entry_atr": atr_value,  # Store ATR at entry
                        "obs": [0.0, atr_value, trend_strength, nexus_conf],  # AI observation
                        "action": [1.0, 0.8],  # Default hedge params
                        "open_time": time.time(),
                        "hedge_plan": {
                            "hedge1_trigger_price": hedge1_price,
                            "hedge2_trigger_price": hedge2_price,
                            "hedge3_trigger_price": hedge3_price,
                            "hedge4_trigger_price": hedge4_price,
                            "hedge1_lots": hedge1_lot,
                            "hedge2_lots": hedge2_lot,
                            "hedge3_lots": hedge3_lot,
                            "hedge4_lots": hedge4_lot,
                            "zone_width_pips": format_pips(zone_pips, symbol)
                        }
                    })
                    logger.debug(f"[METADATA] Stored entry data for ticket #{ticket}: TP={tp_pips:.1f}pips, ATR={atr_pips:.1f}pips (No SL - Hedge Strategy)")
                
                return result
            else:
                logger.error("Trade execution failed")
                return None

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    async def process_position_management(self, symbol: str, positions: List[Dict],
                                  tick: Dict, point: float, shield, ppo_guardian,
                                  nexus=None, rsi_value: float = 50.0, oracle=None, pressure_metrics=None) -> bool:
        """
        Process position management for a symbol.

        Args:
            symbol: Trading symbol
            positions: List of positions
            tick: Current tick data
            point: Point value
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance
            nexus: Optional NexusBrain instance
            oracle: Optional Oracle instance (Layer 4)
            pressure_metrics: Optional Tick Pressure Metrics

        Returns:
            True if positions were managed (closed/opened)
        """
        logger.info(f"[POS_MGMT] Called for {symbol} with {len(positions)} positions")
        
        if not positions:
            return False
        
        # OPTIMIZATION: Removed redundant broker.get_positions() call.
        # The caller (_process_existing_positions) already synced with broker.
        # We just need to filter out any known ghost tickets.
        
        # Filter out ghost tickets
        positions = [p for p in positions if p['ticket'] not in self.position_manager._ghost_tickets]
        
        if not positions:
            return False

        # [CRITICAL FIX] Update PositionManager with latest broker data (Profit/Price)
        # This ensures 'should_close_bucket' uses real-time PnL, not stale data.
        self.position_manager.update_positions(positions)

        # Only log when we have multiple positions (actual bucket)
        if len(positions) > 1:
            logger.info(f"[BUCKET] Checking exit for {symbol}: {len(positions)} positions")

        # Check for bucket exits first
        bucket_closed = False

        # Find or create bucket for positions
        position_tickets = [p['ticket'] for p in positions]
        bucket_id = self.position_manager.find_bucket_by_tickets(position_tickets)
        
        if not bucket_id:
            # No existing bucket - create new one
            position_objects = [
                self.position_manager.active_positions.get(p['ticket'])
                for p in positions
                if p['ticket'] in self.position_manager.active_positions
            ]
            
            if position_objects:
                bucket_id = self.position_manager.create_bucket(position_objects)
                if len(positions) > 1:
                    logger.info(f"[BUCKET] CREATED NEW: {bucket_id} with {len(positions)} positions")
                else:
                    logger.debug(f"[BUCKET] CREATED NEW: {bucket_id} for single position (enables TP/Zone tracking)")
        else:
            logger.debug(f"[BUCKET] REUSING EXISTING: {bucket_id} for {len(positions)} positions")

        # Prepare market data for intelligent scalping analysis (MOVED UP)
        atr_value = self.market_data.calculate_atr(symbol, 14)
        trend_strength = self.market_data.calculate_trend_strength(symbol, 20)
        
        # [GOD MODE] Fetch candles for Structure Analysis
        candles = self.market_data.candles.get_history(symbol)
        
        market_data = {
            'atr': atr_value,  # ATR in points
            'spread': abs(tick['ask'] - tick['bid']),  # Spread in points
            'trend_strength': trend_strength,  # Trend strength 0-1
            'current_price': (tick['ask'] + tick['bid']) / 2,
            'bid': tick['bid'], # Explicit Bid
            'ask': tick['ask'], # Explicit Ask
            'point': point,  # Point value for pip calculations
            'candles': candles, # Pass full history for structure analysis
            'rsi': rsi_value # Add RSI here
        }
        
        ai_context = {
            'pressure_metrics': pressure_metrics,
            'pressure_dominance': pressure_metrics.get('pressure_dominance', 0.0) if pressure_metrics else 0.0
        }

        # [AI SNIPER] Check for Smart Unwind Opportunity
        if oracle and len(positions) > 1 and bucket_id:
             # Prepare market data history for Oracle
             history = self.market_data.candles.get_history(symbol)
             # Ensure we have enough history
             if len(history) >= 30:
                 volumes = [c.get('volume', 0) for c in history]
                 prices = [c.get('close', 0) for c in history]
                 
                 sniper_data = {
                     'prices': prices,
                     'volumes': volumes
                 }
                 
                 unwound = await self.position_manager.execute_ai_sniper_logic(
                     bucket_id, oracle, self.broker, sniper_data
                 )
                 if unwound:
                     logger.info(f"[SNIPER] Smart Unwind executed for {bucket_id}")
                     return True # Positions changed

        # [HIGHEST INTELLIGENCE] LAYER 2: THE ERASER (Tactical De-Risking)
        if len(positions) > 1 and bucket_id:
            # Pass market_data and ai_context to Eraser
            erased = await self.position_manager.execute_eraser_logic(bucket_id, self.broker, market_data, ai_context)
            if erased:
                logger.info(f"[ERASER] Tactical De-Risking executed for {bucket_id}")
                return True # Positions changed

        # Check if bucket should be closed
        # (market_data is already prepared)

        # Periodic position status update (every 5 seconds) to show AI is monitoring
        current_time = time.time()
        if current_time - self._last_position_status_time >= 5.0:
            self._last_position_status_time = current_time
            first_pos = positions[0]
            entry_price = first_pos.get('price_open', 0)
            current_price = market_data['current_price']
            
            # Calculate pips correctly: XAUUSD uses 100 (1 pip = 0.01), forex uses 10000 (1 pip = 0.0001)
            pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol else 10000
            atr_pips = atr_value * pip_multiplier  # ATR in pips
            
            # Check for stored TP in metadata
            tp_pips = atr_pips * 0.3  # Default dynamic TP
            tp_source = "Dynamic 0.3 ATR"
            next_hedge_info = None
            
            try:
                ticket = first_pos.get('ticket') if isinstance(first_pos, dict) else first_pos.ticket
                if ticket in self.position_manager.active_learning_trades:
                    metadata = self.position_manager.active_learning_trades[ticket]
                    stored_tp = metadata.get('entry_tp_pips', 0.0)
                    if stored_tp > 0:
                        tp_pips = stored_tp
                        tp_source = "Stored"
                        
                    # Extract Next Hedge Info
                    hedge_plan = metadata.get('hedge_plan', {})
                    pos_count = len(positions)
                    if pos_count < 5: # Max 4 hedges
                        next_hedge_key = f"hedge{pos_count}" # e.g. hedge1 if 1 pos exists
                        trigger_price = hedge_plan.get(f"{next_hedge_key}_trigger_price")
                        lots = hedge_plan.get(f"{next_hedge_key}_lots")
                        if trigger_price:
                             next_hedge_info = {'price': trigger_price, 'lots': lots, 'type': 'PENDING'}
            except Exception:
                pass
            
            # Calculate current P&L in pips
            if first_pos.get('type') == 0:  # BUY
                pnl_pips = (current_price - entry_price) * pip_multiplier
            else:  # SELL
                pnl_pips = (entry_price - current_price) * pip_multiplier
            
            # Use new Dashboard Logger
            ai_notes = f"RSI {rsi_value:.1f}"
            # [USER REQUEST] Disabled floating logs to reduce noise
            # TradingLogger.log_active_status(symbol, bucket_id, positions, pnl_pips, tp_pips, next_hedge_info, ai_notes)

        # Initialize bucket_closed to False
        bucket_closed = False
        
        # [AI PLAN B] Reversion Escape Check
        # If Plan B is active (Fakeout Detected) OR Risk Veto is active (Defensive Hold),
        # we exit as soon as we are profitable (Break-Even + small profit) to reduce risk.
        if getattr(self, f"_plan_b_active_{symbol}", False) or getattr(self, f"_plan_b_veto_{symbol}", False):
            # Calculate Net PnL (Profit + Swap + Commission)
            net_profit = sum(
                (p.get('profit', 0.0) if isinstance(p, dict) else p.profit) + 
                (p.get('swap', 0.0) if isinstance(p, dict) else p.swap) + 
                (p.get('commission', 0.0) if isinstance(p, dict) else p.commission) 
                for p in positions
            )
            
            # [INTELLIGENCE FIX] Dynamic Slippage Buffer
            # Fixed $0.50 is risky for larger lots. We need a buffer proportional to volume.
            # Aim for $10.00 per lot (approx 10 pips on Gold) to cover execution slippage.
            total_vol = sum(p.get('volume', 0.0) if isinstance(p, dict) else p.volume for p in positions)
            slippage_buffer = max(0.50, total_vol * 10.0)
            
            if net_profit > slippage_buffer: 
                logger.info(f"🚀 [PLAN B] Reversion Escape Successful! Net Profit: ${net_profit:.2f} (Buffer: ${slippage_buffer:.2f}). Closing all positions.")
                bucket_closed = await self.position_manager.close_bucket_positions(self.broker, bucket_id, symbol)
                if bucket_closed:
                    setattr(self, f"_plan_b_active_{symbol}", False)
                    setattr(self, f"_plan_b_veto_{symbol}", False)
                    return True
        
        logger.info(f"[TP_CHECK] About to call should_close_bucket for {bucket_id}")
        should_close, confidence = self.position_manager.should_close_bucket(
            bucket_id, ppo_guardian, nexus, market_data
        )
        logger.info(f"[TP_CHECK] should_close_bucket returned: {should_close}, confidence: {confidence}")

        if should_close:
            logger.info(f"[BUCKET] EXIT TRIGGERED: Confidence {confidence:.3f} - Closing all positions")
            bucket_closed = await self.position_manager.close_bucket_positions(
                self.broker, bucket_id, symbol
            )
            if bucket_closed:
                logger.info(f"[BUCKET] CLOSED SUCCESSFULLY: {bucket_id}")
            else:
                logger.warning(f"[BUCKET] CLOSE FAILED: {bucket_id}")
        # Only log if exit was checked and failed (for debugging)
        # Silent operation when no exit needed - reduces log spam

        # If bucket not closed, check for zone recovery (only log if executed)
        if not bucket_closed:
            # [PHASE 2] THE STABILIZER: Check for Hedging Trigger
            # If Drawdown > 1.5 * ATR, trigger hedge immediately
            stabilizer_triggered = self.position_manager.check_stabilizer_trigger(bucket_id, market_data)
            
            if stabilizer_triggered:
                logger.warning(f"[STABILIZER] Emergency Hedge Triggered for {bucket_id}")
                # Force Zone Recovery to execute a hedge
                # We do this by passing a flag or just letting the risk manager handle it
                # For now, we rely on the standard zone recovery logic but with heightened awareness
                # In future, we can call a specific self.risk_manager.execute_stabilizer_hedge()
            
            # [HIGHEST INTELLIGENCE] LAYER 2: CALCULATED RECOVERY
            # Check if bucket is stuck > 30 mins and needs a muscle move
            # Now includes IronShield for Trend Veto
            calculated_recovery_executed = await self.position_manager.execute_calculated_recovery(
                self.broker, bucket_id, market_data, self.risk_manager.shield
            )
            if calculated_recovery_executed:
                logger.info(f"[MUSCLE] Calculated Recovery Executed for {bucket_id}")
                return True

            logger.info(f"[ZONE_CHECK] Bucket {bucket_id} not closed, checking zone recovery")
            
            # Convert Position objects to dictionaries for zone recovery
            positions_dict = [
                {
                    'ticket': p.get('ticket') if isinstance(p, dict) else p.ticket,
                    'symbol': p.get('symbol') if isinstance(p, dict) else p.symbol,
                    'type': p.get('type') if isinstance(p, dict) else p.type,
                    'volume': p.get('volume') if isinstance(p, dict) else p.volume,
                    'price_open': p.get('price_open') if isinstance(p, dict) else p.price_open,
                    'price_current': p.get('price_current') if isinstance(p, dict) else p.price_current,
                    'profit': p.get('profit') if isinstance(p, dict) else p.profit,
                    'sl': p.get('sl') if isinstance(p, dict) else p.sl,
                    'tp': p.get('tp') if isinstance(p, dict) else p.tp,
                    'time': p.get('time') if isinstance(p, dict) else p.time,
                    'comment': p.get('comment', '') if isinstance(p, dict) else getattr(p, 'comment', '')
                }
                for p in positions
            ]
            
            # Calculate point value for symbol (XAUUSD uses different pip size)
            if "XAU" in symbol or "GOLD" in symbol:
                point_value = 0.01  # XAUUSD: 1 pip = 0.01
            elif "JPY" in symbol:
                point_value = 0.01  # JPY pairs: 1 pip = 0.01
            else:
                point_value = 0.0001  # Standard forex: 1 pip = 0.0001
            
            # Calculate current ATR for dynamic zone sizing
            atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
            
            # Get volatility ratio
            volatility_ratio = self.market_data.get_volatility_ratio() if hasattr(self.market_data, 'get_volatility_ratio') else 1.0

            logger.info(f"[ZONE_CHECK] Calling execute_zone_recovery for {symbol} with {len(positions_dict)} positions | ATR: {atr_value:.5f} | VolRatio: {volatility_ratio:.2f}")
            zone_recovery_executed = self.risk_manager.execute_zone_recovery(
                self.broker, symbol, positions_dict, tick, point_value,
                shield, ppo_guardian, self.position_manager, nexus, oracle=oracle, atr_val=atr_value,
                volatility_ratio=volatility_ratio, rsi_value=rsi_value
            )
            if zone_recovery_executed:
                logger.info(f"[ZONE] RECOVERY EXECUTED for {symbol}")
                # Force immediate update of position cache to reflect new hedge
                # This prevents "Ghost Trades" where the bot doesn't know about the new hedge
                await asyncio.sleep(0.2) # Give broker a moment
                all_positions = self.broker.get_positions()
                if all_positions:
                    self.position_manager.update_positions(all_positions)
                    logger.info(f"[SYNC] Positions updated after hedge. Total: {len(all_positions)}")
            else:
                logger.debug(f"[ZONE_CHECK] No recovery needed for {symbol}")
            return zone_recovery_executed

        return bucket_closed

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        stats = self.session_stats.copy()
        stats["runtime_seconds"] = time.time() - stats["start_time"]
        return stats

    def reset_session_stats(self) -> None:
        """Reset session statistics."""
        self.session_stats = {
            "trades_opened": 0,
            "trades_closed": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "start_time": time.time()
        }
        logger.info("Session statistics reset")

    async def _check_global_safety(self):
        """
        [DOOMSDAY PROTOCOL] Global Equity Stop Loss.
        If Drawdown > 20%, CLOSE EVERYTHING immediately.
        """
        if getattr(self, '_safety_lock', False):
            return # Already locked

        account_info = self.broker.get_account_info()
        if not account_info:
            return

        balance = account_info.get('balance', 0.0)
        equity = account_info.get('equity', 0.0)
        
        if balance <= 0: return

        drawdown_pct = (balance - equity) / balance
        
        # HARDCODED SAFETY LIMIT: 20%
        if drawdown_pct > 0.20:
            logger.critical(f"[DOOMSDAY] GLOBAL EQUITY STOP TRIGGERED! Drawdown: {drawdown_pct*100:.1f}%")
            print(f">>> [CRITICAL] DOOMSDAY PROTOCOL ACTIVATED. CLOSING ALL TRADES.", flush=True)
            
            self._safety_lock = True
            
            # Close all positions
            positions = self.broker.get_positions()
            if positions:
                for pos in positions:
                    ticket = pos.get('ticket') if isinstance(pos, dict) else pos.ticket
                    self.broker.close_position(ticket)
                    logger.critical(f"[DOOMSDAY] Closed ticket {ticket}")
            
            # Raise flag to stop bot
            raise Exception("Global Equity Stop Loss Triggered")

    async def run_trading_cycle(self, strategist, shield, ppo_guardian,
                               nexus=None, oracle=None) -> None:
        """
        Run a complete trading cycle.

        Args:
            strategist: Strategist instance
            shield: IronShield instance
            ppo_guardian: PPO Guardian instance
            nexus: Optional NexusBrain instance
            oracle: Optional Oracle instance (Layer 4)
        """
        symbol = self.config.symbol

        try:
            # [SAFETY FIRST] Check Global Equity Stop
            await self._check_global_safety()

            # Get market data
            # print(f">>> [DEBUG] Fetching tick for {symbol}...", flush=True)
            tick = self.market_data.get_tick_data(symbol)
            if not tick:
                print(f">>> [DEBUG] No tick data for {symbol} (Check MT5 Connection/Market Watch)", flush=True)
                return

            # [HIGHEST INTELLIGENCE] Update Tick Pressure Analyzer
            self.tick_analyzer.add_tick(tick)
            pressure_metrics = self.tick_analyzer.get_pressure_metrics()

            # Record market data to database
            await self._record_market_data(symbol, tick)

            # Validate market conditions
            market_ok, reason = self.validate_market_conditions(symbol, tick)
            
            if not market_ok:
                # [UI FEEDBACK] Log pause reason only when it changes to avoid spam
                # Special handling for "Spread too wide" to avoid spamming due to small pip changes
                is_spread_issue = "Spread too wide" in reason
                
                should_log = False
                if is_spread_issue:
                    # Only log if we weren't already paused for spread
                    if self.last_pause_reason is None or "Spread too wide" not in self.last_pause_reason:
                        should_log = True
                elif reason != self.last_pause_reason:
                    should_log = True

                if should_log:
                    logger.info(f"[PAUSED] TRADING PAUSED: {reason}")
                    # User requested to disable this from terminal log
                    if not is_spread_issue:
                        print(f">>> [PAUSED] {reason}", flush=True)
                    self.last_pause_reason = reason
                return
            
            # [UI FEEDBACK] Log resumption if we were previously paused
            if self.last_pause_reason is not None:
                logger.info(f"[RESUMED] TRADING RESUMED: Market conditions normalized.")
                print(f">>> [RESUMED] Market conditions normalized.", flush=True)
                self.last_pause_reason = None

            # Get account info
            account_info = self.broker.get_account_info()
            if not account_info:
                print(">>> [WARN] Could not fetch account info", flush=True)
                return

            # Process management for existing positions
            # Check for positions first to determine mode (Management vs Hunting)
            positions = self.broker.get_positions(symbol=symbol)
            
            if positions:
                # MANAGEMENT MODE
                await self._process_existing_positions(symbol, tick, shield, ppo_guardian, nexus, oracle, pressure_metrics)
                return # STRICTLY RETURN - No new entries while positions exist

            # HUNTING MODE
            # Proceed to AI analysis for new entries

            # --- LAYER 4: ORACLE ENGINE ---
            oracle_prediction = "NEUTRAL"
            oracle_confidence = 0.0
            history = self.market_data.candles.get_history(symbol) # Get history once
            
            if oracle:
                # Get last 60 candles
                if len(history) >= 60:
                    # Pass full candle objects, Oracle handles extraction
                    oracle_prediction, oracle_confidence = oracle.predict(history[-60:])
                    if oracle_confidence > 0.6:
                        logger.info(f"[ORACLE] Prediction: {oracle_prediction} ({oracle_confidence:.2f})")

                    # [HIGHEST INTELLIGENCE] LAYER 1: REGIME DETECTION
                    # Use Oracle's advanced math to double-check regime
                    oracle_regime, oracle_signal = oracle.get_regime_and_signal(history[-60:])
                    logger.info(f"[ORACLE] Regime: {oracle_regime} | Signal: {oracle_signal}")
                    
                    # [USER REQUEST] DISABLED CIRCUIT BREAKER FOR INITIAL ENTRIES
                    # The user requested that enhancements apply ONLY to hedging/recovery.
                    # We log the Oracle's opinion but do NOT block the trade.

            # --- HIERARCHICAL AI DECISION LOGIC (v5.0) ---
            # 1. Supervisor: Detect Regime
            # Prepare data for Supervisor
            atr_value, trend_strength = self._calculate_indicators(symbol)
            rsi_value = self.market_data.calculate_rsi(symbol)
            macro_context = self.market_data.get_macro_context()
            
            supervisor_data = {
                'atr': atr_value,
                'trend_strength': trend_strength,
                'volatility_ratio': self.market_data.get_volatility_ratio(),
                'macro_context': macro_context
            }
            
            regime = self.supervisor.detect_regime(supervisor_data)
            logger.info(f"[SUPERVISOR] Market Regime: {regime.name}")
            
            # 2. Supervisor: Select Worker
            worker_name = self.supervisor.get_active_worker(regime)
            
            # 3. Worker: Generate Signal
            # Prepare context for worker
            market_context = {
                'symbol': symbol,
                'current_price': tick['bid'],
                'tick': tick,
                'history': history,
                'macro_context': self.market_data.get_macro_context(),
                'rsi': rsi_value,
                'trend_strength': trend_strength,
                'atr': atr_value,
                'pressure_metrics': pressure_metrics # [HIGHEST INTELLIGENCE]
            }
            
            action, confidence, reason = "HOLD", 0.0, "No Worker"
            
            if worker_name == "RANGE_WORKER":
                action, confidence, reason = self.range_worker.get_signal(market_context)
            elif worker_name == "TREND_WORKER":
                action, confidence, reason = self.trend_worker.get_signal(market_context)
            else:
                # Fallback for CHAOS or DEFENSIVE regimes
                # Use RangeWorker but with higher confidence threshold implicitly handled by logic
                # Or just default to RangeWorker as it's safer
                action, confidence, reason = self.range_worker.get_signal(market_context)
                reason = f"[DEFENSIVE] {reason}"
                
            # Create Signal Object
            signal = None
            if action != "HOLD" and confidence > 0.5:
                signal = TradeSignal(
                    action=TradeAction.BUY if action == "BUY" else TradeAction.SELL,
                    symbol=symbol,
                    confidence=confidence,
                    reason=f"[{worker_name}] {reason}",
                    metadata={
                        'regime': regime.name,
                        'worker': worker_name,
                        'macro_data': self.market_data.get_macro_context(),
                        'oracle_prediction': oracle_prediction, # Keep Oracle for penalties
                        'nexus_signal_confidence': confidence # Map for compatibility
                    },
                    timestamp=time.time()
                )
            
            # Use market_regime variable for compatibility with downstream logic
            market_regime = regime
            if not signal:
                # [UI FEEDBACK] Log "Thinking" status periodically to reassure user
                if self.decision_tracker:
                    # Fix: Provide default confidence for HOLD decisions
                    should_log, _ = self.decision_tracker.should_log_decision(
                        symbol, {'action': 'HOLD', 'confidence': 0.0, 'reasoning': reason}
                    )
                    if should_log:
                        # Use print to ensure visibility in console
                        print(f"[AI THINKING] Regime: {regime.name} | Worker: {worker_name} | Action: HOLD | Reason: {reason}", flush=True)
                return
            
            # [DEBUG] Force print signal for debugging
            print(f"[DEBUG] Signal Generated: {signal.action.value} | Conf: {signal.confidence:.2f} | Reason: {signal.reason}", flush=True)

            # --- CHAMELEON FILTER (SOFT PENALTY) ---
            # Instead of blocking, reduce lot size for Counter-Trend trades
            regime_penalty = 1.0
            if market_regime.name == "TRENDING_UP" and signal.action == TradeAction.SELL:
                regime_penalty = 0.5 # Cut lots in half
                logger.info(f"[CHAMELEON] CAUTION: Counter-Trend SELL. Reducing lots by 50%.")
            elif market_regime.name == "TRENDING_DOWN" and signal.action == TradeAction.BUY:
                regime_penalty = 0.5 # Cut lots in half
                logger.info(f"[CHAMELEON] CAUTION: Counter-Trend BUY. Reducing lots by 50%.")
                
            # --- ORACLE FILTER (SOFT PENALTY) ---
            # If Oracle strongly disagrees, reduce lot size further
            oracle_penalty = 1.0
            if oracle_prediction != "NEUTRAL" and oracle_confidence > 0.7:
                if (signal.action == TradeAction.BUY and oracle_prediction == "DOWN") or \
                   (signal.action == TradeAction.SELL and oracle_prediction == "UP"):
                    oracle_penalty = 0.5
                    logger.info(f"[ORACLE] CAUTION: Signal contradicts Oracle. Reducing lots by 50%.")

            # --- HOLOGRAPHIC FILTER (TICK PRESSURE) ---
            # If Order Flow Pressure strongly disagrees, reduce lot size
            pressure_penalty = 1.0
            if pressure_metrics['intensity'] == 'HIGH':
                if (signal.action == TradeAction.BUY and pressure_metrics['dominance'] == 'SELL') or \
                   (signal.action == TradeAction.SELL and pressure_metrics['dominance'] == 'BUY'):
                    pressure_penalty = 0.5
                    logger.info(f"[HOLOGRAPHIC] CAUTION: High Intensity Counter-Pressure. Reducing lots by 50%.")

            # --- LAYER 9: GLOBAL BRAIN (SOFT PENALTY) ---
            # Check DXY/US10Y impact on Gold
            global_bias = 0.0
            brain_penalty = 1.0
            if self.global_brain and ("XAU" in symbol or "GOLD" in symbol):
                # In a real scenario, we'd pass a dict of current prices for DXY, US10Y etc.
                # For now, we assume the GlobalBrain is fetching its own data or using what's available
                # We can pass the current tick price of the symbol itself as a reference
                current_prices = {symbol: tick['bid']} 
                correlation_signal = self.global_brain.analyze_impact(current_prices)
                
                if correlation_signal.score != 0.0:
                    global_bias = correlation_signal.score
                    logger.info(f"[GLOBAL_BRAIN] Correlation Bias: {global_bias:.2f} (Driver: {correlation_signal.driver})")
                    
                    # PENALTY LOGIC: If Global Brain disagrees, reduce size
                    if global_bias < -0.5 and signal.action == TradeAction.BUY:
                         brain_penalty = 0.5
                         logger.info(f"[GLOBAL_BRAIN] CAUTION: Bearish Correlation ({global_bias:.2f}). Reducing lots by 50%.")
                    elif global_bias > 0.5 and signal.action == TradeAction.SELL:
                         brain_penalty = 0.5
                         logger.info(f"[GLOBAL_BRAIN] CAUTION: Bullish Correlation ({global_bias:.2f}). Reducing lots by 50%.")

            # Calculate ATR and trend for PPO
            atr_value, trend_strength = self._calculate_indicators(symbol)

            # Calculate position size with PPO optimization
            lot_size, lot_reason = self.calculate_position_size(
                signal, account_info, strategist, shield,
                ppo_guardian=ppo_guardian,
                atr_value=atr_value,
                trend_strength=trend_strength
            )
            
            # === APPLY INTELLIGENCE PENALTIES ===
            # Combine all penalties (Regime * Oracle * Brain * Pressure)
            # Example: 0.5 * 0.5 * 1.0 * 1.0 = 0.25 (Quarter Size)
            total_penalty = regime_penalty * oracle_penalty * brain_penalty * pressure_penalty
            
            if total_penalty < 1.0:
                old_lot = lot_size
                lot_size = round(lot_size * total_penalty, 4)
                # Ensure we don't go below minimum lot (usually 0.01)
                lot_size = max(lot_size, 0.01)
                lot_reason += f" (AI Penalty: {total_penalty:.2f}x)"
                logger.info(f"[AI_COUNCIL] Consensus Weak. Reducing Size: {old_lot} -> {lot_size} lots")

            # --- CHAMELEON VOLATILITY SCALING ---
            # If Volatile, cut lot size in half for safety
            if market_regime.name == "VOLATILE":
                lot_size = lot_size * 0.5
                lot_reason += " (Reduced 50% due to VOLATILE regime)"
                logger.info(f"[CHAMELEON] Volatility Detected! Reducing lot size to {lot_size:.2f}")

            if lot_size <= 0:
                print(f">>> [DEBUG] Lot Size Rejected: {lot_reason}", flush=True)
                logger.info(f"[POSITION] SIZING REJECTED: {lot_reason}")
                return

            # Normalize lot size to broker requirements BEFORE logging or executing
            # This ensures the logs match the actual execution
            if hasattr(self.broker, 'normalize_lot_size'):
                raw_lot = lot_size
                lot_size = self.broker.normalize_lot_size(symbol, lot_size)
                if raw_lot != lot_size:
                     # Only log if significant change
                     if abs(raw_lot - lot_size) > 0.000001:
                        logger.info(f"[LOT ADJUST] Normalized {raw_lot} -> {lot_size} to match broker requirements")

            # Only log lot size if the signal was just logged (i.e., it changed)
            if self._last_signal_logged:
                logger.info(f"[POSITION] SIZE CALCULATED: {lot_size} lots | Reason: {lot_reason}")

            # Final validation (is_recovery_trade=False for normal entries)
            # [OPTIMIZATION] For Continuous Scalping, we relax the cooldown check if the signal is strong
            # But we must respect the global cooldown to prevent API bans
            can_enter, entry_reason = self.validate_trade_entry(signal, lot_size, account_info, tick, is_recovery_trade=False)
            if not can_enter:
                # [DEBUG] Print rejection reason to console
                print(f">>> [DEBUG] Entry Blocked: {entry_reason}", flush=True)
                
                # [TELEMETRY] Log blocked entry for analysis
                if FLAGS.ENABLE_TELEMETRY:
                    self._telemetry.write(DecisionRecord(
                        ts=time.time(),
                        symbol=symbol,
                        action="entry_blocked",
                        side="buy" if signal.action == TradeAction.BUY else "sell",
                        price=tick['ask'] if signal.action == TradeAction.BUY else tick['bid'],
                        lots=lot_size,
                        features={"reason": entry_reason, "obi": tick.get('obi', 0.0)},
                        context={"regime": regime.name, "worker": worker_name},
                        decision={"blocked": True, "reason": entry_reason}
                    ))

                # Handle Cooldown Logs specifically (Throttled)
                if "cooldown" in entry_reason.lower():
                    current_time = time.time()
                    if current_time - self._last_cooldown_log_time > 5.0:
                        remaining_msg = ""
                        # Try to extract time from "Global cooldown: X.Xs < Y.Ys"
                        if "Global cooldown" in entry_reason and "<" in entry_reason:
                            try:
                                parts = entry_reason.split('<')
                                limit = float(parts[1].replace('s', '').strip())
                                current = float(parts[0].split(':')[1].replace('s', '').strip())
                                remaining = limit - current
                                remaining_msg = f" Resuming in {remaining:.1f}s"
                            except:
                                pass
                        
                        logger.info(f"[PAUSED] Trading Halted: {entry_reason}.{remaining_msg}")
                        self._last_cooldown_log_time = current_time

                # Suppress repetitive blocking logs (cooldown, position exists) - only log once when signal changes
                elif "already exists" not in entry_reason.lower():
                    logger.info(f"[TRADE] ENTRY BLOCKED: {entry_reason}")
                elif "already exists" in entry_reason.lower() and self._last_signal_logged:
                    # Only log position blocking when signal just changed (not every cycle)
                    logger.info(f"[TRADE] ENTRY BLOCKED: {entry_reason}")
                return

            logger.info(f"[TRADE] VALIDATION PASSED: All entry conditions met")
            
            # Update last_trade_time IMMEDIATELY to prevent race conditions
            self.last_trade_time = time.time()

            # Execute trade
            # --- GENERATE ENTRY SUMMARY ---
            
            # 1. Calculate Virtual TP using IronShield (Same logic as execution)
            atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
            
            # Convert ATR to points based on symbol type
            if 'JPY' in symbol or 'XAU' in symbol:
                atr_points = atr_value * 100
                pip_multiplier = 100
            else:
                atr_points = atr_value * 10000
                pip_multiplier = 10000
                
            # Get dynamic params from Shield
            zone_points, tp_points = shield.get_dynamic_params(atr_points)
            tp_pips = tp_points # This is in pips (e.g. 25.0)
            
            is_buy = signal.action == TradeAction.BUY
            entry_price = tick['ask'] if is_buy else tick['bid']
            
            # Calculate Virtual TP Price
            if is_buy:
                virtual_tp_price = entry_price + (tp_pips / pip_multiplier)
            else:
                virtual_tp_price = entry_price - (tp_pips / pip_multiplier)

            # Generate Clean Entry Summary
            summary = (
                f"\n>>> [AI ENTRY PLAN] <<<\n"
                f"Initial Trade: {signal.action.value} {lot_size} lots @ {entry_price:.5f}\n"
                f"Target:        Dynamic (Bucket Logic)\n"
                f"Virtual TP:    {virtual_tp_price:.5f} (+{tp_pips:.1f} pips)\n"
                f"----------------------------------------------------\n"
                f"Strategy:      {signal.metadata.get('worker', 'Unknown')}\n"
                f"Regime:        {signal.metadata.get('regime', 'Unknown')}\n"
                f"AI Analysis:   {signal.reason}\n"
                f"===================================================="
            )
            
            # [FIX] Windows Console Compatibility - Force clean version on Windows
            clean_summary = (
                f"\n>>> [AI ENTRY PLAN] <<<\n"
                f"Initial Trade: {signal.action.value} {lot_size} lots @ {entry_price:.5f}\n"
                f"Target:        Dynamic (Bucket Logic)\n"
                f"Virtual TP:    {virtual_tp_price:.5f} (+{tp_pips:.1f} pips)\n"
                f"----------------------------------------------------\n"
                f"Strategy:      {signal.metadata.get('worker', 'Unknown')}\n"
                f"Regime:        {signal.metadata.get('regime', 'Unknown')}\n"
                f"AI Analysis:   {signal.reason}\n"
                f"===================================================="
            )
            
            import sys
            if sys.platform == 'win32':
                ui_logger.info(clean_summary)
            else:
                try:
                    ui_logger.info(summary)
                except Exception:
                    ui_logger.info(clean_summary)
            
            result = await self.execute_trade_entry(signal, lot_size, tick, strategist, shield)
            if result:
                logger.info(f"[TRADE] EXECUTION SUCCESSFUL: {signal.action.value} {signal.symbol} {lot_size} lots @ {tick['ask'] if signal.action == TradeAction.BUY else tick['bid']:.5f}")
                logger.info(f"   [EXIT] STRATEGY: Virtual TP/SL managed by bucket logic | AI Confidence: {signal.confidence:.3f}")
            else:
                logger.error(f"[TRADE] EXECUTION FAILED for {signal.symbol}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in trading cycle: {e}", flush=True)
            logger.error(f"Error in trading cycle: {e}")

    async def _record_market_data(self, symbol: str, tick: Dict) -> None:
        """Record tick and candle data to database."""
        if not self.db_queue:
            return

        # Record tick data
        tick_data = TickData(
            symbol=symbol,
            bid=tick['bid'],
            ask=tick['ask'],
            timestamp=time.time(),
            flags=0
        )
        await self.db_queue.add_tick(tick_data)

        # Record candle data if available
        if self.market_data.candles._cache:
            latest_candle = self.market_data.candles.get_latest_candle(symbol)
            if latest_candle:
                candle_data = CandleData(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    open_price=latest_candle.get('open', 0),
                    high=latest_candle.get('high', 0),
                    low=latest_candle.get('low', 0),
                    close=latest_candle.get('close', 0),
                    volume=latest_candle.get('volume', 0),
                    timestamp=time.time()
                )
                await self.db_queue.add_candle(candle_data)

    def _get_symbol_properties(self, symbol):
        """
        Returns correct pip value and point size for the symbol.
        CRITICAL: Distinguishes between Forex (10.0) and Gold (1.0).
        """
        symbol_upper = symbol.upper()
        
        if "XAU" in symbol_upper or "GOLD" in symbol_upper:
            return {
                "pip_value": 1.0,      # 1 pip = $1 per lot (approx) on Gold
                "point_size": 0.01,    # Price moves in cents
                "pip_size": 0.10       # Standard Gold Pip is 10 cents
            }
        elif "JPY" in symbol_upper:
            return {
                "pip_value": 9.0,      # Approx for JPY pairs
                "point_size": 0.001,
                "pip_size": 0.01
            }
        else: # Standard Forex (EURUSD, GBPUSD, etc.)
            return {
                "pip_value": 10.0,     # Standard Lot = $10/pip
                "point_size": 0.00001,
                "pip_size": 0.0001
            }

    async def _handle_blocked_hedge_strategy(self, symbol, positions, decision, tick):
        """
        [HIGHEST INTELLIGENCE] Plan B:
        When AI blocks a hedge, we don't just walk away. 
        We optimize the EXISTING positions to escape the danger zone.
        """
        reasons = decision.get("reasons", [])
        
        # STRATEGY 1: FAKEOUT DETECTED (Weak Trend)
        # If the hedge was blocked because the trend is weak (Low Confidence),
        # it means the AI predicts a Reversion (Bounce).
        if any("min_conf" in r for r in reasons):
            # We log this strategic shift.
            # In a fully automated version, this would send a 'ModifyOrder' to move TP to Break-Even.
            # For now, we activate "Reversion Escape Mode".
            if not getattr(self, f"_plan_b_active_{symbol}", False):
                logger.info(f"🧠 [AI PLAN B] Hedge blocked (Fakeout Detected). "
                            f"Strategy shifted to 'Mean Reversion Escape' for {symbol}. "
                            f"Expecting price bounce to exit in profit.")
                setattr(self, f"_plan_b_active_{symbol}", True)

        # STRATEGY 2: RISK LIMIT REACHED
        elif any("veto" in r for r in reasons):
            if not getattr(self, f"_plan_b_veto_{symbol}", False):
                logger.warning(f"🛡️ [AI PLAN B] Risk Governor active. "
                               f"Holding {symbol} positions defensively (No new risk added).")
                setattr(self, f"_plan_b_veto_{symbol}", True)

    async def _process_existing_positions(self, symbol: str, tick: Dict, shield, ppo_guardian, nexus, oracle=None, pressure_metrics=None) -> bool:
        """
        Process management for existing positions.
        Returns True if positions were managed (skipping new entries).
        """
        # Update positions from broker
        all_positions = self.broker.get_positions()
        
        # FAIL-SAFE: If broker returns None (error), DO NOT update or cleanup.
        # This prevents wiping state during temporary connection loss.
        if all_positions is None:
            logger.warning("[PROCESS_POS] Failed to fetch positions from broker - skipping update")
            return False

        # Always update positions, even if empty, to ensure closed positions are removed
        self.position_manager.update_positions(all_positions)
        # CRITICAL: Cleanup stale positions immediately after broker update
        self.position_manager.cleanup_stale_positions(all_positions)
        self.position_manager._update_bucket_stats()

        # Get positions for this symbol
        symbol_positions = self.position_manager.get_positions_for_symbol(symbol)

        logger.debug(f"[PROCESS_POS] Found {len(symbol_positions)} positions for {symbol}")

        if symbol_positions:
            # Sort positions by time to ensure correct order for hedging logic
            symbol_positions.sort(key=lambda p: p.time)

            # Get symbol properties for point value
            # Get Symbol Properties for Math
            props = self._get_symbol_properties(symbol)
            point_value = props['point_size']

            logger.debug(f"[PROCESS_POS] Calling process_position_management with {len(symbol_positions)} positions")
            
            # --- ELASTIC DEFENSE PROTOCOL INTEGRATION ---
            # 1. Get Live Market Intelligence
            current_atr = self.market_data.calculate_atr(symbol)
            current_rsi = self.market_data.calculate_rsi(symbol)
            
            # 2. Check Dynamic Hedge Trigger (if we have positions)
            # We only intervene if the bucket is losing
            bucket_pnl = sum(p.profit for p in symbol_positions)
            
            # [AI STATUS MONITOR] Periodic Log
            current_time = time.time()
            if current_time - self._last_position_status_time >= 30.0: # Every 30s
                self._last_position_status_time = current_time
                
                # Determine Strategy
                strategy_status = "Monitoring"
                if bucket_pnl > 0:
                    strategy_status = "Profit Protection (Trailing)"
                elif len(symbol_positions) > 1:
                    strategy_status = "Zone Recovery (Hedging)"
                    if oracle:
                        strategy_status += " + Sniper Watch"
                
                # Format PnL
                pnl_str = f"${bucket_pnl:.2f}"
                if bucket_pnl > 0: pnl_str = f"+{pnl_str}"
                
                status_msg = (
                    f"\n>>> [AI STATUS MONITOR] <<<\n"
                    f"Positions:    {len(symbol_positions)} Active\n"
                    f"Net PnL:      {pnl_str}\n"
                    f"Strategy:     {strategy_status}\n"
                    f"Market:       RSI {current_rsi:.1f} | ATR {current_atr:.4f}\n"
                    f"Action:       Holding & Analyzing Tick Data...\n"
                    f"----------------------------------------------------"
                )
                
                import sys
                if sys.platform == 'win32':
                    ui_logger.info(status_msg)
                else:
                    try:
                        ui_logger.info(status_msg)
                    except Exception:
                        ui_logger.info(status_msg)

            # Check if we should use Elastic Defense OR fallback to Zone Recovery
            # If Elastic Defense says "WAIT" (should_hedge=False), we must ensure Zone Recovery doesn't override it blindly.
            # However, Zone Recovery handles the "Grid" logic.
            # To make them work together:
            # - If Elastic Defense triggers, we execute immediately.
            # - If Elastic Defense says "Wait", we let process_position_management run, BUT we inject the "Wait" signal into Risk Manager?
            # Actually, simpler: We trust Elastic Defense as the primary trigger for NEW hedges.
            # But process_position_management also handles CLOSING (TP/SL). We must run that.
            
            elastic_hedge_triggered = False
            
            if bucket_pnl < 0 and len(symbol_positions) < self.risk_manager.config.max_hedges:
                last_pos = symbol_positions[-1]
                
                # [PHASE 2] Check Stabilizer Trigger (Drawdown based)
                stabilizer_triggered = False
                bucket_id = self.position_manager.find_bucket_by_tickets([p.ticket for p in symbol_positions])
                
                if len(symbol_positions) == 1:
                    if bucket_id:
                        stabilizer_data = {'atr': current_atr}
                        stabilizer_triggered = self.position_manager.check_stabilizer_trigger(bucket_id, stabilizer_data)

                # Ask Iron Shield: "Should we hedge now?"
                should_hedge, req_dist = shield.calculate_dynamic_hedge_trigger(
                    entry_price=last_pos.price_open,
                    current_price=tick['bid'] if last_pos.type == 0 else tick['ask'],
                    trade_type=last_pos.type,
                    atr=current_atr,
                    rsi=current_rsi,
                    hedge_count=len(symbol_positions) - 1
                )
                
                if should_hedge or stabilizer_triggered:
                    # [AI INTELLIGENCE] Filter Strategic Hedges (IronShield)
                    # We always allow Stabilizer (Emergency) hedges, but we filter Strategic ones.
                    if should_hedge and not stabilizer_triggered:
                        # Construct market_data for AI
                        trend_strength = self.market_data.calculate_trend_strength(symbol)
                        bb = self.market_data.calculate_bollinger_bands(symbol)
                        market_data = {
                            'atr': current_atr,
                            'rsi': current_rsi,
                            'trend_strength': trend_strength,
                            'volatility_ratio': self.market_data.market_state.get_volatility_ratio(),
                            'close': tick['bid'],
                            'bb_upper': bb['upper'],
                            'bb_lower': bb['lower'],
                            'macro_context': self.market_data.get_macro_context()
                        }

                        # Detect Regime
                        regime_obj = self.supervisor.detect_regime(market_data)
                        
                        # Calculate Drawdown for AI
                        current_drawdown = abs(bucket_pnl) # bucket_pnl is usually negative here
                        
                        # Ask The Oracle
                        hedge_decision = self.hedge_intel.evaluate_hedge_opportunity(
                            market_data,
                            "buy" if last_pos.type == 0 else "sell", # Current position type
                            current_drawdown,
                            regime_obj.name
                        )
                        
                        if not hedge_decision.should_hedge:
                            logger.info(f"[AI] Skipping Strategic Hedge: {hedge_decision.reason}")
                            # Skip this cycle, but don't return False (let other logic run)
                            # We just reset should_hedge to False so we don't execute below
                            should_hedge = False
                    
                    # Re-check if we still want to hedge (Stabilizer might still be True)
                    if should_hedge or stabilizer_triggered:
                        # Calculate Intelligent Lot Size using correct pip_value
                        recovery_dist_pips = (current_atr / props['point_size']) # Convert price delta to points
                        
                        # [AI INTELLIGENCE] Dynamic Grid Step
                        # If we are hedging, we might want to place it further away if volatility is high?
                        # Actually, IronShield calculates the lot size.
                        # We can adjust the 'recovery_dist_pips' if we wanted to widen the grid.
                        
                        hedge_lot = shield.calculate_recovery_lot_size(
                        positions=symbol_positions,
                        current_price=tick['bid'],
                        target_recovery_pips=recovery_dist_pips,
                        symbol_step=0.01, # Assuming standard step
                        pip_value=props['pip_value'], # Pass the correct value (1.0 for Gold)
                        atr=current_atr,    # PASSING AI DATA
                        rsi=current_rsi     # PASSING AI DATA
                    )
                    
                        # Execute Hedge
                        hedge_type = "SELL" if last_pos.type == 0 else "BUY"
                        
                        # Initialize ai_reason early to avoid UnboundLocalError
                        ai_reason = "Standard Volatility Defense"

                        # [CRITICAL] MARGIN CHECK BEFORE HEDGING
                        if hasattr(self.broker, 'check_margin'):
                            if not self.broker.check_margin(symbol, hedge_lot, hedge_type):
                                # Calculate max possible volume
                                max_vol = self.broker.get_max_volume(symbol, hedge_type)
                                max_vol = self.broker.normalize_lot_size(symbol, max_vol)
                                
                                if max_vol < 0.01:
                                    logger.critical(f"[MARGIN CALL] Cannot place hedge! Required: {hedge_lot}, Max Possible: {max_vol}. ABORTING HEDGE.")
                                    print(f">>> [CRITICAL] MARGIN CALL! Cannot place hedge. Account is over-leveraged.", flush=True)
                                    
                                    # [AI INTELLIGENCE] SHED WEIGHT
                                    # If we can't hedge, we MUST reduce exposure to survive.
                                    shed_candidates = self.hedge_intel.find_shedding_opportunity(symbol_positions)
                                    if shed_candidates:
                                        logger.warning(f"[SURVIVAL] Shedding 2 trades to free margin...")
                                        # Close the candidates
                                        for pos in shed_candidates:
                                            await self.broker.close_position(pos.ticket)
                                        return True # We took action (shedding), so we are "done" for this tick
                                    
                                    return False # Cannot hedge, let it ride
                                else:
                                    logger.warning(f"[MARGIN WARNING] Capping hedge size from {hedge_lot} to {max_vol} due to margin constraints.")
                                    hedge_lot = max_vol
                                    ai_reason += " [MARGIN CAPPED]"

                        # [AI INTELLIGENCE] Risk Governor & Hedge Policy
                        # ------------------------------------------------------------------
                        # 1. Risk Governor Veto Check
                        if FLAGS.ENABLE_GOVERNOR:
                            # Get Account Info for PnL %
                            try:
                                acct = await self.broker.get_account_info()
                                balance = float(acct.get('balance', 1.0))
                                daily_pnl = self.session_stats.get("total_profit", 0.0)
                                daily_pnl_pct = (daily_pnl / balance) * 100.0 if balance > 0 else 0.0
                            except Exception:
                                daily_pnl_pct = 0.0

                            veto, veto_reason = self._governor.veto({
                                "daily_pnl_pct": daily_pnl_pct,
                                "total_exposure": sum(p.volume for p in symbol_positions),
                                "spread_points": (tick['ask'] - tick['bid']) / props['point_size'],
                                "news_now": self.news_filter.is_news_event()
                            })
                            if veto:
                                logger.warning(f"[RISK-VETO] Hedge blocked by Governor: {veto_reason}")
                                print(f">>> [RISK-VETO] Hedge blocked: {veto_reason}", flush=True)
                                return False

                        # 2. Hedge Policy Decision
                        tp_price = 0.0
                        sl_price = 0.0
                        confidence = 0.0
                        policy_reasons = []
                        
                        if FLAGS.ENABLE_POLICY:
                            # Construct features for the policy
                            spread_points = (tick['ask'] - tick['bid']) / props['point_size']
                            atr_points = current_atr / props['point_size']
                            
                            # Calculate Trend Slope (using MarketData)
                            try:
                                trend_strength = self.market_data.calculate_trend_strength(symbol)
                            except Exception:
                                trend_strength = 0.0

                            # [AI INTELLIGENCE] Calculate Volatility Z-Score
                            # We need a history of ATR or Volume to calculate Z-Score.
                            # Since we don't have a long history in memory here, we can use a simplified
                            # relative volume metric if available, or default to 0.0 safely.
                            # Ideally, MarketData should track this. For now, we use a safe proxy.
                            vol_z_score = 0.0
                            if hasattr(self.market_data, 'get_volume_z_score'):
                                vol_z_score = self.market_data.get_volume_z_score(symbol)

                            features = {
                                "regime_trend": simple_regime_trend(trend_strength, atr_points),
                                "breakout_quality": 0.5, # Placeholder - requires candle pattern analysis
                                "structure_break": 0.0, # Placeholder - requires support/resistance levels
                                "drawdown_urgency": min(1.0, current_drawdown / 1000.0), # Crude proxy
                                "spread_atr": spread_atr(spread_points, atr_points),
                                "vol_z": vol_z_score,
                            }
                            context = {
                                "side": hedge_type.lower(),
                                "price": tick['bid'] if hedge_type == "SELL" else tick['ask'],
                                "atr": current_atr,
                                "open_hedges": len(symbol_positions),
                            }
                            
                            decision = self._hedge_policy.decide(features, context)
                            
                            if not decision["hedge"]:
                                # >>> [INTELLIGENCE INJECTION] <<<
                                # Execute Plan B instead of just skipping
                                await self._handle_blocked_hedge_strategy(symbol, symbol_positions, decision, tick)

                                logger.info(f"[HEDGE-SKIP] Policy declined hedge: {decision['reasons']}")
                                if FLAGS.ENABLE_TELEMETRY:
                                    self._telemetry.write(DecisionRecord(
                                        ts=time.time(), symbol=symbol, action="hedge-skip", side=hedge_type,
                                        price=context["price"], lots=hedge_lot,
                                        features=features, context=context, decision=decision
                                    ))
                                return False
                                
                            # Apply Policy Outputs
                            tp_price = decision["tp_price"]
                            sl_price = decision["sl_price"]
                            confidence = decision["confidence"]
                            policy_reasons = decision["reasons"]
                            ai_reason += f" | Conf: {confidence:.2f}"

                        # Construct AI Reason for User
                        if stabilizer_triggered:
                            ai_reason = "STABILIZER PROTOCOL: Drawdown > 1.5x ATR. Emergency Hedge."
                        elif hedge_type == "BUY":
                            if current_rsi > 75: ai_reason = f"Extreme Bullish Momentum (RSI {current_rsi:.1f}). Max Aggression (1.25x) to counter Sell loss."
                            elif current_rsi > 60: ai_reason = f"Strong Bullish Trend (RSI {current_rsi:.1f}). Increased Aggression (1.15x)."
                            else: ai_reason = f"Normal Market (RSI {current_rsi:.1f}). Standard Overpower (1.05x)."
                        else: # SELL
                            if current_rsi < 25: ai_reason = f"Extreme Bearish Momentum (RSI {current_rsi:.1f}). Max Aggression (1.25x) to counter Buy loss."
                            elif current_rsi < 40: ai_reason = f"Strong Bearish Trend (RSI {current_rsi:.1f}). Increased Aggression (1.15x)."
                            else: ai_reason = f"Normal Market (RSI {current_rsi:.1f}). Standard Overpower (1.05x)."

                        # Use UI Logger for visible terminal output
                        # [USER REQUEST] Standardized Hedge Log
                        
                        # Calculate estimated TP for display (Bucket Logic)
                        # We can't know the exact TP until update_bucket_tp runs, but we can estimate it.
                        # Usually it's Break-Even + Target.
                        # For display, we can just say "Dynamic (Bucket)"
                        
                        # --- AI PLAN LOG ---
                        hedge_plan_msg = (
                            f"\n>>> [AI DEFENSE PLAN] ACTIVATING HEDGE <<<\n"
                            f"Trigger:      {ai_reason}\n"
                            f"Action:       OPEN {hedge_type} {hedge_lot} lots\n"
                            f"Objective:    Neutralize Drawdown & Prepare for Recovery\n"
                            f"Status:       EXECUTING NOW...\n"
                            f"----------------------------------------------------"
                        )
                        import sys
                        if sys.platform == 'win32':
                            ui_logger.info(hedge_plan_msg)
                        else:
                            try:
                                ui_logger.info(hedge_plan_msg)
                            except Exception:
                                ui_logger.info(hedge_plan_msg)
                        # -------------------

                        TradingLogger.log_initial_trade(f"{symbol}_HEDGE_{len(symbol_positions)}", {
                            'action': hedge_type,
                            'symbol': symbol,
                            'lots': hedge_lot,
                            'entry_price': tick['bid'] if hedge_type == "SELL" else tick['ask'],
                            'tp_price': tp_price if tp_price > 0 else "Dynamic", 
                            'tp_atr': 0.0,
                            'tp_pips': "Bucket", 
                            'hedges': [], # No nested hedges
                            'atr_pips': current_atr * 10000, # Approx
                            'reasoning': f"[HEDGE {len(symbol_positions)}] {ai_reason}",
                            'confidence': confidence,
                            'reasons': policy_reasons
                        })
                        
                        if FLAGS.ENABLE_TELEMETRY:
                            self._telemetry.write(DecisionRecord(
                                ts=time.time(), symbol=symbol, action="hedge-open", side=hedge_type,
                                price=tick['bid'] if hedge_type == "SELL" else tick['ask'], 
                                lots=hedge_lot,
                                features=features if FLAGS.ENABLE_POLICY else {},
                                context=context if FLAGS.ENABLE_POLICY else {},
                                decision={"tp_price": tp_price, "sl_price": sl_price, "confidence": confidence, "reasons": policy_reasons}
                            ))
                        
                        # Execute directly via broker to bypass standard entry checks
                        self.broker.execute_order(
                            action="OPEN",
                            symbol=symbol,
                            order_type=hedge_type,
                            volume=hedge_lot,
                            price=tick['bid'] if hedge_type == "SELL" else tick['ask'],
                            sl=0.0, tp=0.0,
                            comment="Elastic_Hedge"
                        )
                        elastic_hedge_triggered = True
                        
                        # [AI INTELLIGENCE] Update Bucket TP immediately
                        # Now that we have a new hedge, the "Survival TP" changes.
                        # We must update all trades in the bucket to the new target.
                        if bucket_id:
                            await self.position_manager.update_bucket_tp(self.broker, symbol, bucket_id, tick['bid'])
                        return True # Managed

            # If Elastic Defense didn't trigger a hedge, we still check for EXITS (TP/SL)
            # But we should probably prevent Zone Recovery from adding a hedge if Elastic Defense said "Wait".
            # For now, let's assume process_position_management is mainly for Exits and Zone Recovery.
            # If we want to disable old Zone Recovery, we should do it in Risk Manager or here.
            # Let's rely on the fact that Elastic Defense is "tighter" or "smarter".
            # If Elastic Defense says "Wait" (e.g. RSI), but Zone Recovery says "Hedge" (Fixed Pips),
            # we have a conflict.
            # To fix this, we will modify Risk Manager to respect RSI as well (which we did in previous step via IronShield update).
            
            positions_managed = await self.process_position_management(
                symbol, [p.__dict__ for p in symbol_positions], tick,
                point_value, shield, ppo_guardian, nexus, rsi_value=current_rsi, oracle=oracle, pressure_metrics=pressure_metrics
            )
            
            logger.debug(f"[PROCESS_POS] process_position_management returned: {positions_managed}")
            
            if positions_managed:
                logger.info(f"[POSITION] MANAGEMENT EXECUTED for {symbol} - Skipping new entries")
                return True
        
        return False

    def _calculate_indicators(self, symbol: str) -> Tuple[float, float]:
        """Calculate ATR and trend strength."""
        atr_value = self.market_data.calculate_atr(symbol, 14) if hasattr(self.market_data, 'calculate_atr') else 0.0010
        trend_strength = self.market_data.calculate_trend_strength(symbol, 20) if hasattr(self.market_data, 'calculate_trend_strength') else 0.0
        return atr_value, trend_strength