"""
Market Data Manager - Handles candle data, tick processing, and market state tracking.

This module provides a clean interface for market data operations, including:
- Candle data fetching and caching
- Tick data processing
- Market regime detection
- Adaptive sleep timing based on market conditions

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from threading import Lock
import MetaTrader5 as mt5

logger = logging.getLogger("MarketDataManager")


class CorrelationMonitor:
    """
    Phase 1 Upgrade: Monitors correlated assets (USDJPY, US500) 
    to provide macro context to the AI.
    """
    def __init__(self, usd_symbol="USDJPY", risk_symbol="US500"):
        self.usd_symbol = usd_symbol
        self.risk_symbol = risk_symbol
        self.initialized = False
        self._initialize_symbols()

    def _initialize_symbols(self):
        """Ensure correlation symbols are available in MT5."""
        for sym in [self.usd_symbol, self.risk_symbol]:
            if not mt5.symbol_select(sym, True):
                logger.warning(f"[WARN] Could not select correlation symbol: {sym}. Macro vision will be limited.")
            else:
                logger.info(f"[MACRO] Macro Eye active: Watching {sym}")
        self.initialized = True

    def get_macro_state(self) -> List[float]:
        """
        Returns a vector representing the macro environment.
        [USD_Velocity, Risk_Velocity]
        Velocity > 0 means Bullish, < 0 means Bearish.
        """
        if not self.initialized:
            return [0.0, 0.0]

        data = {}
        for sym in [self.usd_symbol, self.risk_symbol]:
            # Get last 2 ticks to calculate immediate velocity
            try:
                ticks = mt5.copy_ticks_from(sym, datetime.now(), 10, mt5.COPY_TICKS_ALL)
                if ticks is None or len(ticks) < 2:
                    data[sym] = 0.0
                    continue
                
                # Calculate simple velocity (Price Change)
                # Normalized by price to get percentage
                current = ticks[-1][1] # bid
                prev = ticks[0][1]
                if prev == 0:
                    data[sym] = 0.0
                else:
                    velocity = ((current - prev) / prev) * 10000 # Scaled up for AI
                    data[sym] = velocity
            except Exception as e:
                logger.debug(f"Failed to fetch macro data for {sym}: {e}")
                data[sym] = 0.0

        # Return [USD_Strength, Risk_Sentiment]
        # Note: If USDJPY rises, Dollar is Strong.
        return [data.get(self.usd_symbol, 0.0), data.get(self.risk_symbol, 0.0)]


@dataclass
class MarketRegime:
    """Represents current market conditions and volatility state."""
    regime: str  # "NORMAL", "HIGH_VOL", "PANIC", "LOW_VOL"
    spread_multiplier: float
    sleep_base: float
    should_skip_trading: bool


class MarketStateManager:
    """
    Tracks market volatility and determines trading conditions.
    Provides adaptive behavior based on current market regime.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.spread_history: List[float] = []
        self.price_history: List[float] = []
        self._lock = Lock()

    def update(self, price: float, spread_points: float) -> None:
        """Update market state with new price and spread data."""
        with self._lock:
            self.price_history.append(price)
            self.spread_history.append(spread_points)

            # Keep only recent history
            if len(self.price_history) > self.window_size:
                self.price_history = self.price_history[-self.window_size:]
                self.spread_history = self.spread_history[-self.window_size:]

    def get_regime(self) -> str:
        """Determine current market regime based on volatility."""
        if len(self.spread_history) < 10:
            return "NORMAL"

        with self._lock:
            avg_spread = sum(self.spread_history[-10:]) / 10

            if avg_spread > 50:
                return "PANIC"
            elif avg_spread > 20:
                return "HIGH_VOL"
            elif avg_spread < 5:
                return "LOW_VOL"
            else:
                return "NORMAL"

    def should_skip_trading(self) -> bool:
        """Check if trading should be paused due to extreme volatility."""
        return self.get_regime() == "PANIC"

    def get_adaptive_sleep(self) -> float:
        """Get adaptive sleep time based on market conditions."""
        regime = self.get_regime()

        base_sleep = {
            "PANIC": 5.0,      # Slow down during panic
            "HIGH_VOL": 2.0,   # Moderate slowdown
            "NORMAL": 1.0,     # Normal operation
            "LOW_VOL": 0.5     # Speed up in calm markets
        }.get(regime, 1.0)

        return base_sleep

    def get_volatility_ratio(self) -> float:
        """
        Calculate the ratio of recent volatility to average volatility.
        Returns:
            Ratio > 1.0 means higher than average volatility.
        """
        with self._lock:
            if len(self.price_history) < 20:
                return 1.0
            
            # Calculate returns (absolute price changes)
            prices = self.price_history
            moves = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            
            if not moves:
                return 1.0
                
            # Average move of the last 20 ticks
            avg_move = sum(moves[-20:]) / len(moves[-20:])
            
            if avg_move == 0:
                return 1.0
                
            # Last move
            last_move = moves[-1]
            
            return last_move / avg_move


class CandleManager:
    """
    Manages candle data fetching, caching, and processing.
    Provides different candle formats for various AI components.
    """

    def __init__(self, broker_adapter, timeframe: str = "M1"):
        self.broker = broker_adapter
        self.timeframe = timeframe
        self._cache: Dict[str, Dict] = {}
        self._cache_lock = Lock()
        self._cache_timeout = 30  # seconds

    def get_history(self, symbol: str, force_refresh: bool = False) -> List[Dict]:
        """
        Get recent price history for trading decisions.

        Args:
            symbol: Trading symbol
            force_refresh: Force fresh data from broker

        Returns:
            List of candle dictionaries with OHLC data
        """
        cache_key = f"{symbol}_history"

        # Check cache first
        if not force_refresh:
            with self._cache_lock:
                if cache_key in self._cache:
                    cached_data, timestamp = self._cache[cache_key]
                    if time.time() - timestamp < self._cache_timeout:
                        return cached_data

        try:
            # Fetch from broker
            candles = self.broker.get_market_data(symbol, self.timeframe, 100)
            if not candles:
                logger.warning(f"No candle data received for {symbol}")
                return []

            # Cache the result
            with self._cache_lock:
                self._cache[cache_key] = (candles, time.time())

            return candles

        except Exception as e:
            logger.error(f"Failed to fetch candle history for {symbol}: {e}")
            return []

    def get_full_candles(self, symbol: str) -> List[Dict]:
        """Get full candle data for technical analysis."""
        return self.get_history(symbol)

    def get_nexus_candles(self, symbol: str) -> List[List]:
        """
        Get candle data formatted for NexusBrain predictions.
        Format: [open, high, low, close, volatility]
        CRITICAL: Must match training data features (volatility = high - low)
        """
        candles = self.get_history(symbol)
        # Calculate volatility (High - Low) to match training data
        return [[c['open'], c['high'], c['low'], c['close'], c['high'] - c['low']] for c in candles]

    def get_latest_candle(self, symbol: str) -> Optional[Dict]:
        """
        Get the most recent completed candle.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary containing the latest candle data or None if unavailable
        """
        candles = self.get_history(symbol)
        if candles:
            return candles[-1]
        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._cache_lock:
            self._cache.clear()


class MarketDataManager:
    """
    Main interface for market data operations.
    Coordinates between different data sources and provides unified access.
    """

    def __init__(self, broker_adapter, timeframe: str = "M1", config: Dict = None):
        self.broker = broker_adapter
        self.candles = CandleManager(broker_adapter, timeframe)
        self.market_state = MarketStateManager()
        
        # HFT: Order Book Imbalance Cache
        self.last_obi = 0.0
        self.last_obi_time = 0.0

        # [PHASE 1] Initialize Correlation Monitor
        self.macro_eye = None
        if config and config.get('trading', {}).get('correlations', {}).get('enable_correlation', False):
            corr_config = config['trading']['correlations']
            self.macro_eye = CorrelationMonitor(
                usd_symbol=corr_config.get('usd_proxy', 'USDJPY'),
                risk_symbol=corr_config.get('risk_proxy', 'US500')
            )

    def get_macro_context(self) -> List[float]:
        """
        Get current macro environment state.
        Returns: [USD_Velocity, Risk_Velocity]
        """
        if self.macro_eye:
            return self.macro_eye.get_macro_state()
        return [0.0, 0.0]

    def get_order_book_imbalance(self, symbol: str) -> float:
        """
        HFT Layer 7: Calculates Order Book Imbalance (OBI).
        Returns:
            -1.0 (Strong Sell Pressure) to +1.0 (Strong Buy Pressure)
        """
        try:
            # Fetch Depth of Market (Level 2)
            if hasattr(self.broker, 'get_order_book'):
                book = self.broker.get_order_book(symbol)
                if not book:
                    return 0.0
                
                # Calculate total volume on Bid and Ask sides
                # Assuming book returns {'bids': [{'price': 1.1, 'volume': 10}, ...], 'asks': ...}
                total_bid_vol = sum(item.get('volume', 0) for item in book.get('bids', []))
                total_ask_vol = sum(item.get('volume', 0) for item in book.get('asks', []))
                
                if total_bid_vol + total_ask_vol == 0:
                    return 0.0
                    
                # OBI Formula: (BidVol - AskVol) / (BidVol + AskVol)
                obi = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)
                return obi
            
            return 0.0
        except Exception as e:
            # logger.error(f"OBI Calculation failed: {e}") # Suppress spam
            return 0.0

    def get_symbol_properties(self, symbol: str) -> Dict[str, Any]:
        """
        Get standardized properties for a symbol.
        
        Returns:
            Dict with keys: 'pip_multiplier', 'point_value', 'pip_divisor'
        """
        if "XAU" in symbol or "GOLD" in symbol:
            return {
                'pip_multiplier': 100,
                'point_value': 0.01,
                'pip_divisor': 100
            }
        elif "JPY" in symbol:
            return {
                'pip_multiplier': 100,
                'point_value': 0.01,
                'pip_divisor': 100
            }
        else:
            return {
                'pip_multiplier': 10000,
                'point_value': 0.0001,
                'pip_divisor': 10000
            }

    def get_tick_data(self, symbol: str) -> Optional[Dict]:
        """
        Get current tick data with validation.
        Includes HFT OBI calculation.

        Args:
            symbol: Trading symbol

        Returns:
            Tick data dict or None if failed
        """
        try:
            tick = self.broker.get_tick(symbol)
            if not tick or 'bid' not in tick or 'ask' not in tick:
                logger.warning(f"Invalid tick data received for {symbol}")
                return None

            current_time = time.time()

            # Update market state
            if 'bid' in tick and 'ask' in tick:
                mid_price = (tick['bid'] + tick['ask']) / 2
                spread_points = abs(tick['ask'] - tick['bid'])
                self.market_state.update(mid_price, spread_points)

            # HFT: Update Order Book Imbalance (every 1s to avoid API spam)
            if current_time - self.last_obi_time > 1.0:
                self.last_obi = self.get_order_book_imbalance(symbol)
                self.last_obi_time = current_time
            
            # Inject OBI into tick data for downstream logic
            tick['obi'] = self.last_obi

            return tick

        except Exception as e:
            logger.error(f"Failed to get tick data for {symbol}: {e}")
            return None

    def get_volatility_ratio(self) -> float:
        """Get current volatility ratio from market state."""
        return self.market_state.get_volatility_ratio()

    def validate_market_conditions(self, symbol: str, tick: Dict) -> Tuple[bool, str]:
        """
        Validate if market conditions are suitable for trading.

        Args:
            symbol: Trading symbol
            tick: Current tick data

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check market regime
        if self.market_state.should_skip_trading():
            return False, "Market in panic mode - trading paused"

        # Check spread
        spread_points = abs(tick['ask'] - tick['bid'])
        regime = self.market_state.get_regime()

        max_spread = {
            "PANIC": 100,
            "HIGH_VOL": 50,
            "NORMAL": 30,
            "LOW_VOL": 20
        }.get(regime, 30)

        if spread_points > max_spread:
            return False, f"Spread too wide: {spread_points:.1f} > {max_spread}"

        return True, "Market conditions OK"

    def get_adaptive_sleep_time(self) -> float:
        """Get recommended sleep time based on market conditions."""
        return self.market_state.get_adaptive_sleep()

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            symbol: Trading symbol
            period: Period for ATR calculation (default 14)

        Returns:
            ATR value or 0.0010 if calculation fails
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period + 1:
                logger.warning(f"Insufficient candle data for ATR calculation: {len(candles)} < {period + 1}")
                return 0.0010

            # Calculate True Range for each candle
            true_ranges = []
            for i in range(1, len(candles)):
                current = candles[i]
                previous = candles[i-1]

                # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
                tr1 = current['high'] - current['low']
                tr2 = abs(current['high'] - previous['close'])
                tr3 = abs(current['low'] - previous['close'])

                true_range = max(tr1, tr2, tr3)
                true_ranges.append(true_range)

            # Calculate ATR as simple moving average of True Ranges
            if len(true_ranges) >= period:
                atr = sum(true_ranges[-period:]) / period
                logger.debug(f"[ATR] Calculated ATR for {symbol}: {atr:.6f}")
                return atr
            else:
                logger.warning(f"Insufficient true ranges for ATR: {len(true_ranges)} < {period}")
                return 0.0010

        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0010

    def calculate_trend_strength(self, symbol: str, period: int = 20) -> float:
        """
        Calculate trend strength based on price direction consistency.

        Args:
            symbol: Trading symbol
            period: Period for trend calculation (default 20)

        Returns:
            Trend strength between 0.0 (no trend) and 1.0 (strong trend)
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period:
                logger.warning(f"Insufficient candle data for trend calculation: {len(candles)} < {period}")
                return 0.0

            # Calculate price changes
            price_changes = []
            for i in range(1, min(period + 1, len(candles))):
                change = candles[-i]['close'] - candles[-i-1]['close']
                direction = 1 if change > 0 else -1 if change < 0 else 0
                price_changes.append(direction)

            if not price_changes:
                return 0.0

            # Calculate trend consistency (how many consecutive moves in same direction)
            consistency = 0
            current_streak = 0
            prev_direction = 0

            for direction in price_changes:
                if direction == prev_direction and direction != 0:
                    current_streak += 1
                elif direction != 0:
                    current_streak = 1
                    prev_direction = direction
                else:
                    current_streak = 0
                    prev_direction = 0

                consistency = max(consistency, current_streak)

            # Normalize to 0-1 scale
            trend_strength = min(consistency / period, 1.0)
            logger.debug(f"[TREND] Calculated trend strength for {symbol}: {trend_strength:.3f}")
            return trend_strength

        except Exception as e:
            logger.error(f"Error calculating trend strength for {symbol}: {e}")
            return 0.0

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI).
        Used for momentum filtering in Elastic Defense Protocol.
        """
        try:
            candles = self.candles.get_history(symbol)
            if len(candles) < period + 1:
                return 50.0 # Neutral fallback

            gains = []
            losses = []
            
            for i in range(1, len(candles)):
                change = candles[i]['close'] - candles[i-1]['close']
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            # Simple RSI calculation (not smoothed for speed, but sufficient)
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI for {symbol}: {e}")
            return 50.0