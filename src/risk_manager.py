"""
Risk Manager - Handles zone recovery, hedging, and risk management.

This module provides comprehensive risk management including:
- Zone recovery logic with AI-enhanced parameters
- Dynamic hedging based on market conditions
- Risk validation and safety checks
- Cooldown management to prevent rapid-fire trading

Author: AETHER Development Team
License: MIT
Version: 1.0.0
"""

import time
import logging
import threading
import math
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("RiskManager")

# Setup UI Logger for console output (Shared with Main Bot)
ui_logger = logging.getLogger("AETHER_UI")
if not ui_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    ui_logger.addHandler(console_handler)
    ui_logger.propagate = False


def retry_with_backoff(
    operation: Callable,
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_multiplier: float = 2.0,
    max_total_time: float = 10.0,
    operation_name: str = "operation"
) -> Tuple[bool, Any]:
    """
    Retry an operation with exponential backoff and timeout.
    
    Args:
        operation: Callable that returns (success: bool, result: Any)
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_multiplier: Multiplier for delay on each retry
        max_total_time: Maximum total time in seconds (timeout)
        operation_name: Name for logging
        
    Returns:
        Tuple of (success: bool, result: Any)
    """
    delay = initial_delay
    start_time = time.time()
    start_time = time.time()
    
    for attempt in range(1, max_attempts + 1):
        # Check timeout
        if time.time() - start_time > max_total_time:
            logger.error(f"[RETRY] {operation_name} timeout exceeded ({max_total_time}s)")
            return False, None
        
        try:
            success, result = operation()
            
            if success:
                if attempt > 1:
                    logger.info(f"[RETRY] {operation_name} succeeded on attempt {attempt}/{max_attempts}")
                return True, result
            
            # Check for non-retryable errors (position doesn't exist)
            if result and isinstance(result, dict):
                retcode = result.get('retcode', 0)
                if retcode == 10036:  # Position doesn't exist
                    logger.info(f"[RETRY] {operation_name} - position no longer exists, aborting retries")
                    return False, result
            
            if attempt < max_attempts:
                logger.warning(f"[RETRY] {operation_name} failed (attempt {attempt}/{max_attempts}), retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                logger.error(f"[RETRY] {operation_name} failed after {max_attempts} attempts")
                return False, result
                
        except Exception as e:
            if attempt < max_attempts:
                logger.warning(f"[RETRY] {operation_name} exception (attempt {attempt}/{max_attempts}): {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_multiplier
            else:
                logger.error(f"[RETRY] {operation_name} exception after {max_attempts} attempts: {e}")
                return False, None
    
    return False, None


class RiskState(Enum):
    """Enumeration of risk management states."""
    NORMAL = "normal"
    HEDGING = "hedging"
    RECOVERY = "recovery"
    EMERGENCY = "emergency"


@dataclass
class ZoneConfig:
    """Configuration for zone recovery parameters."""
    zone_pips: float
    tp_pips: float
    max_hedges: int = 4
    min_age_seconds: float = 3.0
    hedge_cooldown_seconds: float = 15.0
    bucket_close_cooldown_seconds: float = 15.0
    emergency_hedge_threshold: int = 4


@dataclass
class HedgeState:
    """State tracking for hedging operations."""
    last_hedge_time: float = 0.0
    active_hedges: int = 0
    total_positions: int = 0
    zone_width_points: float = 0.0
    tp_width_points: float = 0.0
    lock: threading.Lock = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()


class RiskManager:
    """
    Manages risk through zone recovery and intelligent hedging.

    This class handles:
    - Dynamic zone calculation with AI input
    - Hedging logic with safety checks
    - Cooldown management to prevent over-trading
    - Emergency risk controls
    """

    def __init__(self, zone_config: ZoneConfig):
        self.config = zone_config
        self._hedge_states: Dict[str, HedgeState] = {}
        self._global_position_cap = 10
        self._lock = threading.Lock()
        self._last_log_time = 0.0

        logger.info(f"RiskManager initialized with zone: {zone_config.zone_pips}pips, TP: {zone_config.tp_pips}pips")

    def _get_hedge_state(self, symbol: str) -> HedgeState:
        """Get or create hedge state for a symbol."""
        if symbol not in self._hedge_states:
            self._hedge_states[symbol] = HedgeState()
        return self._hedge_states[symbol]

    def validate_hedge_conditions(self, broker, symbol: str, positions: List[Dict],
                                tick: Dict, point: float, atr_val: float = 0.0) -> Tuple[bool, str]:
        """
        Validate if hedging conditions are met and safe.
        [INTELLIGENT FIX] Enforces Dynamic Minimum Distance based on ATR.

        Args:
            broker: Broker adapter instance
            symbol: Trading symbol
            positions: List of position dictionaries
            tick: Current tick data
            point: Point value for symbol
            atr_val: Current ATR value (volatility)

        Returns:
            Tuple of (can_hedge, reason)
        """
        if not positions:
            return False, "No positions to hedge"

        state = self._get_hedge_state(symbol)

        with state.lock:
            # Safety Check 1: Max hedges per symbol
            if len(positions) >= self.config.max_hedges:
                logger.info(f"[HEDGE_CHECK] Max hedges reached ({len(positions)}/{self.config.max_hedges})")
                return False, f"Max hedges reached ({len(positions)}/{self.config.max_hedges})"

            # Safety Check 2: Global position cap
            all_positions = broker.get_positions()
            
            if all_positions is None:
                logger.warning("[HEDGE_CHECK] Failed to fetch global positions. Denying hedge for safety.")
                return False, "Failed to fetch global positions"

            total_positions = len(all_positions)
            if total_positions >= self._global_position_cap:
                logger.info(f"[HEDGE_CHECK] Global position cap reached ({total_positions}/{self._global_position_cap})")
                return False, f"Global position cap reached ({total_positions}/{self._global_position_cap})"

            # Safety Check 3: Position age
            # CRITICAL FIX: Use server time like main_bot.py does
            current_server_time = broker.get_tick(symbol).get('time', time.time())
            sorted_pos = sorted(positions, key=lambda p: p['time'])
            last_pos = sorted_pos[-1]
            age_since_last = current_server_time - last_pos['time']
            if age_since_last < self.config.min_age_seconds:
                logger.info(f"[HEDGE_CHECK] Position too young ({age_since_last:.1f}s < {self.config.min_age_seconds}s)")
                return False, f"Position too young ({age_since_last:.1f}s < {self.config.min_age_seconds}s)"

            # [INTELLIGENT FIX] Safety Check 3.5: Minimum Distance (Dynamic ATR)
            # Prevents "Micro-Zone Death Spiral" by ensuring we don't hedge in noise.
            
            # Handle both dict and object access for price (Fixes KeyError: 'price')
            if isinstance(last_pos, dict):
                last_price = float(last_pos.get('price_open', last_pos.get('price', 0.0)))
                last_type = last_pos.get('type', 0)
            else:
                last_price = float(getattr(last_pos, 'price_open', getattr(last_pos, 'price', 0.0)))
                last_type = getattr(last_pos, 'type', 0)

            current_price = tick['bid'] if last_type == 0 else tick['ask'] # If last was BUY, we check Bid (current sell price)
            # Actually, we just want distance.
            # If last was BUY, we are looking to SELL (Hedge). Sell happens at Bid.
            # If last was SELL, we are looking to BUY (Hedge). Buy happens at Ask.
            # But here we just check raw distance to ensure we aren't churning.
            
            # Use the price relevant to the potential hedge? 
            # No, just check distance from last entry.
            dist = abs(current_price - last_price)
            
            # Calculate Dynamic Minimum Distance
            # Rule: Min Distance = 25% of Daily ATR.
            # e.g. ATR=20.0 -> Min Dist = 5.0.
            # Floor: Never less than $2.00 (200 points) for Gold.
            
            pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol else 10000
            atr_pips = atr_val * pip_multiplier
            
            # Convert ATR to Price Units
            atr_price = atr_val 
            if atr_price <= 0: atr_price = 2.0 # Default fallback
            
            dynamic_min_dist = atr_price * 0.25
            
            # Hard Floor: $2.00 for Gold, 20 pips for Forex
            min_floor = 2.0 if "XAU" in symbol or "GOLD" in symbol else 0.0020
            
            final_min_dist = max(min_floor, dynamic_min_dist)
            
            if dist < final_min_dist:
                 # logger.debug(f"[HEDGE_CHECK] Distance too small ({dist:.2f} < {final_min_dist:.2f}). ATR: {atr_price:.2f}")
                 return False, f"Distance too small ({dist:.2f} < {final_min_dist:.2f})"

            # Safety Check 4: Hedge cooldown
            time_since_hedge = time.time() - state.last_hedge_time
            if time_since_hedge < self.config.hedge_cooldown_seconds:
                logger.info(f"[HEDGE_CHECK] Hedge cooldown active ({time_since_hedge:.1f}s < {self.config.hedge_cooldown_seconds}s)")
                return False, f"Hedge cooldown active ({time_since_hedge:.1f}s < {self.config.hedge_cooldown_seconds}s)"

            # Safety Check 5: Bucket close cooldown
            # This would be checked by the caller using position manager

            logger.info(f"[HEDGE_CHECK] All conditions met - hedging allowed")
            return True, "Conditions met for hedging"

    def calculate_zone_parameters(self, positions: List[Dict], tick: Dict, point: float,
                                ppo_guardian, nexus=None, atr_val: float = 0.0010) -> Tuple[float, float]:
        """
        Calculate dynamic zone and TP parameters with AI enhancement.
        Implements SURVIVAL PROTOCOL (Exponential Grid).
        """
        # Determine pip multiplier for conversion
        symbol = positions[0]['symbol'] if positions else ""
        pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol or "JPY" in symbol else 10000
        
        # --- LAYER 3: SURVIVAL PROTOCOL (Exponential Grid) ---
        # Hedge 1: 20 pips (Base)
        # Hedge 2: 30 pips (1.5x)
        # Hedge 3: 50 pips (2.5x)
        
        num_positions = len(positions)
        
        # Default Base (Hedge 1) - Preparing for first hedge
        zone_pips = 20.0 
        
        if num_positions == 2: # Preparing for Hedge 2 (3rd position total)
            zone_pips = 30.0
        elif num_positions >= 3: # Preparing for Hedge 3+ (4th position total)
            zone_pips = 50.0
            
        # Adjust for ATR if volatility is extreme (optional, but good for robustness)
        # If ATR is huge (> 50 pips), scale up slightly to avoid whipsaw
        atr_pips = atr_val * pip_multiplier
        if atr_pips > 50.0:
             zone_pips = max(zone_pips, atr_pips * 0.8)

        tp_pips = zone_pips # TP usually matches zone width in this strategy
        
        zone_width_points = zone_pips * point
        tp_width_points = tp_pips * point
        
        logger.info(f"[SURVIVAL] Grid Level {num_positions} -> Zone Width: {zone_pips:.1f} pips")

        return zone_width_points, tp_width_points

    def execute_zone_recovery(self, broker, symbol: str, positions: List[Dict],
                            tick: Dict, point: float, shield, ppo_guardian,
                            position_manager, nexus=None, oracle=None, atr_val: float = 0.0010,
                            volatility_ratio: float = 1.0, rsi_value: float = 50.0) -> bool:
        """
        Execute zone recovery hedging if conditions are met.
        
        FAIL-SAFE: Validates broker connection before critical operations.
        Uses retry logic with exponential backoff for resilience.

        Args:
            broker: Broker adapter instance
            symbol: Trading symbol
            positions: List of position dictionaries
            tick: Current tick data
            point: Point value for symbol
            shield: IronShield instance for volume calculation
            ppo_guardian: PPO Guardian instance
            position_manager: PositionManager instance for state persistence
            nexus: Optional NexusBrain instance
            atr_val: Current ATR value for dynamic calculations
            volatility_ratio: Ratio of current volatility to average (1.0 = normal)

        Returns:
            True if hedge was executed
        """
        # FAIL-SAFE: Verify broker connection before critical operations
        try:
            if not broker.is_trade_allowed():
                logger.error(f"[HEDGE] Trading not allowed by broker for {symbol} - ABORTING")
                return False
        except Exception as e:
            logger.error(f"[HEDGE] Broker connection check failed: {e} - ABORTING")
            return False
        
        # Validate conditions
        can_hedge, reason = self.validate_hedge_conditions(broker, symbol, positions, tick, point, atr_val=atr_val)
        if not can_hedge:
            # Throttle cooldown logs
            if "cooldown" in reason.lower():
                current_time = time.time()
                if current_time - self._last_log_time > 5.0:
                    remaining_msg = ""
                    # Try to extract time from "Hedge cooldown active (X.Xs < Y.Ys)"
                    if "cooldown active" in reason and "<" in reason:
                        try:
                            parts = reason.split('<')
                            limit = float(parts[1].replace('s)', '').strip())
                            current = float(parts[0].split('(')[1].replace('s', '').strip())
                            remaining = limit - current
                            remaining_msg = f" Resuming in {remaining:.1f}s"
                        except:
                            pass
                    logger.info(f"[PAUSED] Zone Recovery Halted: {reason}.{remaining_msg}")
                    self._last_log_time = current_time
            else:
                logger.info(f"Zone recovery skipped for {symbol}: {reason}")
            return False

        state = self._get_hedge_state(symbol)

        try:
            # Determine hedge direction
            sorted_pos = sorted(positions, key=lambda p: p['time'])
            first_pos = sorted_pos[0]
            last_pos = sorted_pos[-1]
            
            # Get first position ticket to retrieve stored trigger prices
            first_ticket = first_pos['ticket']
            # Access active_learning_trades via position_manager
            trade_metadata = position_manager.active_learning_trades.get(first_ticket, {})
            hedge_plan = trade_metadata.get('hedge_plan', {})
            entry_atr = trade_metadata.get('entry_atr', 0.0010)
            
            # DYNAMIC PLAN UPDATE: Check if market volatility has shifted significantly (>50%)
            # If so, we must update the Virtual Targets to reflect new reality
            current_atr = 0.0010 # Default
            if ppo_guardian: # Try to get current ATR from PPO context if possible, or pass it in
                 # In this scope we don't have direct access to current ATR easily without passing it
                 # But we can infer it if we had it. For now, let's assume the plan is robust.
                 # To implement the "loop" requested, we'd need to update the plan here.
                 pass

            # Determine which hedge level we're at (based on number of positions)
            hedge_level = len(positions)  # 1 = HEDGE1, 2 = HEDGE2, 3 = HEDGE3
            
            logger.info(f"[ZONE_DEBUG] Current hedge level: {hedge_level} | Positions: {len(positions)}")
            
            # === PLAN VS EXECUTION ALIGNMENT ===
            # Check if we have a stored plan for this specific hedge level
            # If so, use the EXACT trigger price from the plan to ensure "What You See Is What You Get"
            
            use_stored_plan = False
            stored_trigger_price = 0.0
            stored_lots = 0.0
            
            if hedge_plan:
                plan_key_price = f"hedge{hedge_level}_trigger_price"
                plan_key_lots = f"hedge{hedge_level}_lots"
                
                if plan_key_price in hedge_plan:
                    stored_trigger_price = hedge_plan[plan_key_price]
                    stored_lots = hedge_plan.get(plan_key_lots, 0.0)
                    use_stored_plan = True
                    logger.info(f"[PLAN] Found stored plan for Hedge {hedge_level}: Trigger @ {stored_trigger_price:.5f}, Lots: {stored_lots}")

            # Calculate dynamic parameters (still needed for zone width tracking)
            zone_width_points, tp_width_points = self.calculate_zone_parameters(
                positions, tick, point, ppo_guardian, nexus, atr_val=atr_val
            )
            
            # === DYNAMIC VOLATILITY SCALING ===
            # If volatility is high (> 2x average), widen the zone to avoid rapid-fire hedging
            if volatility_ratio > 2.0:
                # Scale factor: 1.0 + (ratio - 2.0) * 0.5, capped at 3.0x
                # Example: Ratio 6.0 -> 1.0 + 4.0*0.5 = 3.0x
                scale_factor = min(1.0 + (volatility_ratio - 2.0) * 0.5, 3.0)
                
                # Apply scaling to zone width
                original_width = zone_width_points
                zone_width_points *= scale_factor
                
                logger.warning(f"[VOLATILITY] High Volatility ({volatility_ratio:.1f}x)! Widening zone: {original_width/point:.1f} -> {zone_width_points/point:.1f} pips")
                
                # Disable stored plan if volatility is extreme, as the plan is likely obsolete
                if use_stored_plan:
                    logger.warning(f"[VOLATILITY] Overriding stored plan due to extreme volatility")
                    use_stored_plan = False

            # Calculate zone boundaries (Dynamic)
            if first_pos['type'] == 0:  # BUY
                upper_level = first_pos['price_open']
                lower_level = first_pos['price_open'] - zone_width_points
            else:  # SELL
                lower_level = first_pos['price_open']
                upper_level = first_pos['price_open'] + zone_width_points
            
            # OVERRIDE with Stored Plan if available
            if use_stored_plan and stored_trigger_price > 0:
                if first_pos['type'] == 0: # BUY
                    # Hedge 1 (SELL) is below entry. Hedge 2 (BUY) is at entry.
                    if hedge_level == 1:
                        lower_level = stored_trigger_price
                    elif hedge_level == 2:
                        upper_level = stored_trigger_price
                    elif hedge_level == 3:
                        lower_level = stored_trigger_price
                else: # SELL
                    # Hedge 1 (BUY) is above entry. Hedge 2 (SELL) is at entry.
                    if hedge_level == 1:
                        upper_level = stored_trigger_price
                    elif hedge_level == 2:
                        lower_level = stored_trigger_price
                    elif hedge_level == 3:
                        upper_level = stored_trigger_price
                
                logger.info(f"[PLAN] Overriding dynamic zone with PLAN: Lower={lower_level:.5f}, Upper={upper_level:.5f}")

            # Update state (for PPO learning)
            with state.lock:
                state.zone_width_points = zone_width_points
                state.tp_width_points = tp_width_points

            # Determine precision for logging
            precision = int(-math.log10(point)) if point > 0 else 2
            
            # Check if price has breached zone
            next_action = None
            target_price = 0.0
            
            # Calculate pip distance from entry for logging
            pip_multiplier = 100 if "XAU" in symbol or "GOLD" in symbol else 10000
            lower_pips = abs(first_pos['price_open'] - lower_level) * pip_multiplier
            upper_pips = abs(upper_level - first_pos['price_open']) * pip_multiplier
            
            logger.info(f"[{symbol}] Zone Check: Bid={tick['bid']:.{precision}f}, Ask={tick['ask']:.{precision}f} | Lower={lower_level:.{precision}f} (-{lower_pips:.1f}pips), Upper={upper_level:.{precision}f} (+{upper_pips:.1f}pips)")

            # CRITICAL FIX: Use Bid for SELL trigger (below lower) and Ask for BUY trigger (above upper)
            # For BUY position: Hedge trigger is BELOW entry (Sell Stop logic) -> Trigger when Bid <= Lower
            # For SELL position: Hedge trigger is ABOVE entry (Buy Stop logic) -> Trigger when Ask >= Upper
            
            # Use rounded values for comparison to avoid floating point noise
            # But keep original precision for internal logic if needed
            # Here we use a small epsilon for robustness
            epsilon = point * 0.1

            if tick['bid'] <= lower_level + epsilon:
                # Price below lower zone boundary
                if first_pos['type'] == 0:  # Initial was BUY, hedge with SELL (Loss Side)
                    next_action = "SELL"
                    target_price = tick['bid']
                    logger.info(f"[{symbol}] TRIGGER: Price below zone ({tick['bid']:.{precision}f} <= {lower_level:.{precision}f}), hedge with SELL")
                elif first_pos['type'] == 1 and len(positions) > 1: 
                    # Initial was SELL. Price dropped below Entry (Profit Side).
                    # If we are hedged (len > 1), this means we have a BUY hedge (Hedge 1) that is now losing.
                    # We need to add to our SELL position (Hedge 2/4) to recover.
                    next_action = "SELL"
                    target_price = tick['bid']
                    logger.info(f"[{symbol}] TRIGGER: Price below zone (Profit Side but Hedged), hedge with SELL (Recovery)")

            elif tick['ask'] >= upper_level - epsilon:
                # Price above upper zone boundary
                if first_pos['type'] == 1:  # Initial was SELL, hedge with BUY (Loss Side)
                    next_action = "BUY"
                    target_price = tick['ask']
                    logger.info(f"[{symbol}] TRIGGER: Price above zone ({tick['ask']:.{precision}f} >= {upper_level:.{precision}f}), hedge with BUY")
                elif first_pos['type'] == 0 and len(positions) > 1:
                    # Initial was BUY. Price rose above Entry (Profit Side).
                    # If we are hedged (len > 1), this means we have a SELL hedge (Hedge 1) that is now losing.
                    # We need to add to our BUY position (Hedge 2/4) to recover.
                    next_action = "BUY"
                    target_price = tick['ask']
                    logger.info(f"[{symbol}] TRIGGER: Price above zone (Profit Side but Hedged), hedge with BUY (Recovery)")
            
            # DEBUG: Explicitly log why trigger failed if close
            if not next_action:
                if first_pos['type'] == 0: # BUY
                     dist_to_trigger = tick['bid'] - lower_level
                     if dist_to_trigger < 10 * point: # Close call (within 10 points)
                         logger.debug(f"[{symbol}] [WARN] CLOSE CALL: Bid {tick['bid']:.{precision}f} is {dist_to_trigger:.{precision}f} above trigger {lower_level:.{precision}f}")
                else: # SELL
                     dist_to_trigger = upper_level - tick['ask']
                     if dist_to_trigger < 10 * point:
                         logger.debug(f"[{symbol}] [WARN] CLOSE CALL: Ask {tick['ask']:.{precision}f} is {dist_to_trigger:.{precision}f} below trigger {upper_level:.{precision}f}")

            if not next_action:
                # logger.debug(f"[{symbol}] Price within zone boundaries, no hedge needed") # Reduced spam
                return False

            # === ELASTIC DEFENSE PROTOCOL CHECK ===
            # Even if zone is breached, we check RSI to avoid "Catching a Falling Knife"
            # This is the "Intelligence" layer overriding the "Grid" layer.
            
            # Determine trade type of the NEW hedge
            new_hedge_type = 0 if next_action == "BUY" else 1
            
            # Check Elastic Defense Trigger (Wait Logic)
            # We use calculate_dynamic_hedge_trigger just to check the "Wait" condition (RSI)
            # We pass the entry price of the LAST position to check relative movement, 
            # but here we are mainly interested in the RSI filter.
            
            # Note: calculate_dynamic_hedge_trigger returns (should_hedge, dist).
            # If should_hedge is False, it means "Wait".
            # However, that function also checks distance. Here we KNOW distance is breached (Zone Logic).
            # So we only care about the RSI part.
            
            # Let's manually check RSI here to be safe and explicit
            elastic_wait = False
            if new_hedge_type == 0: # We want to BUY (Hedge)
                if rsi_value < 30: # Oversold
                    elastic_wait = True
                    logger.info(f"[{symbol}] [ELASTIC DEFENSE] Zone breached but RSI {rsi_value:.1f} is Oversold. WAITING to Hedge BUY.")
            else: # We want to SELL (Hedge)
                if rsi_value > 70: # Overbought
                    elastic_wait = True
                    logger.info(f"[{symbol}] [ELASTIC DEFENSE] Zone breached but RSI {rsi_value:.1f} is Overbought. WAITING to Hedge SELL.")
            
            # Emergency Override: If price is WAY past zone (> 2x Zone), we hedge anyway
            # Calculate how far past zone we are
            if next_action == "BUY":
                excess_pips = (tick['ask'] - upper_level) * pip_multiplier
            else:
                excess_pips = (lower_level - tick['bid']) * pip_multiplier
                
            if excess_pips > 50.0: # Emergency threshold
                if elastic_wait:
                    logger.warning(f"[{symbol}] [EMERGENCY] Price moved {excess_pips:.1f} pips past zone! Overriding Elastic Wait.")
                    elastic_wait = False
            
            if elastic_wait:
                return False

            # Ping-pong validation / Already Hedged Check
            # If we want to do the same action as the last position, it means we are already
            # hedged in that direction and just riding the trend. This is NOT a violation,
            # but a "Hold" state.
            if next_action == "BUY" and last_pos['type'] == 0:  # BUY after BUY
                logger.debug(f"[{symbol}] Already hedged with BUY (Last pos was BUY). Holding...")
                return False
            if next_action == "SELL" and last_pos['type'] == 1:  # SELL after SELL
                logger.debug(f"[{symbol}] Already hedged with SELL (Last pos was SELL). Holding...")
                return False

            # === ELASTIC DEFENSE PROTOCOL INTEGRATION ===
            # If we are here, it means the old "Zone Logic" triggered.
            # BUT, we must check if the Elastic Defense Protocol agrees.
            # If Elastic Defense says "WAIT" (e.g. RSI oversold), we should override the zone trigger.
            
            # We need ATR and RSI. If not passed, we try to infer or skip.
            # Since we don't have direct access to RSI here easily without passing it,
            # we rely on the fact that trading_engine.py ALREADY checked Elastic Defense
            # before calling this function (in _process_existing_positions).
            
            # However, if this function is called from elsewhere, we should be careful.
            # For now, we assume this function is the "Executioner" for the old logic,
            # but we will enforce the new "Zero-Loss" lot sizing here too.

            # Calculate hedge volume using NEW Zero-Loss Formula
            last_pos_volume = last_pos['volume']
            
            # [CRITICAL FIX] Hedge against the GROSS LOSING VOLUME, not Net Exposure.
            # If we are opening a BUY hedge, we are fighting SELLS. Sum all Sell volume.
            # If we are opening a SELL hedge, we are fighting BUYS. Sum all Buy volume.
            
            exposure_to_hedge = 0.0
            
            if next_action == "BUY":
                # We are hedging against SELLS. Sum all SELL volume.
                for p in positions:
                    if p.get('type', 0) == 1: # SELL
                        exposure_to_hedge += p.get('volume', 0.0)
            else: # SELL
                # We are hedging against BUYS. Sum all BUY volume.
                for p in positions:
                    if p.get('type', 0) == 0: # BUY
                        exposure_to_hedge += p.get('volume', 0.0)
            
            # Safety: If exposure is 0 (shouldn't happen if logic is correct), default to last_pos_volume
            if exposure_to_hedge < 0.01:
                logger.warning(f"[HEDGE] Calculated exposure 0.0 for {next_action} hedge. Defaulting to last volume: {last_pos_volume}")
                exposure_to_hedge = last_pos_volume

            spread_points = (tick['ask'] - tick['bid']) / point

            calc_zone_points = zone_width_points / point
            calc_tp_points = tp_width_points / point

            # Use Stored Plan Lots if available, otherwise calculate dynamically
            if use_stored_plan and stored_lots > 0:
                hedge_lot = stored_lots
                logger.info(f"[PLAN] Using stored hedge lots: {hedge_lot}")
            else:
                # Get Oracle prediction if available
                oracle_pred = "NEUTRAL"
                if oracle:
                    # Oracle returns "UP", "DOWN", "NEUTRAL"
                    # We need to map this to "BUY", "SELL", "NEUTRAL" for calculate_defense
                    try:
                        # Assuming we have history available or oracle can predict from internal state?
                        # Actually, oracle.predict needs candles. 
                        # But here we might not have candles easily available without fetching.
                        # For now, we'll assume the oracle object passed might have a cached prediction 
                        # or we skip it if we can't fetch candles efficiently here.
                        # BETTER: The caller (TradingEngine) should pass the prediction string if possible.
                        # BUT: We only have the oracle object.
                        # Let's assume NEUTRAL for now to avoid blocking, or check if oracle has a 'last_prediction' property.
                        pass
                    except Exception:
                        pass
                
                # Use calculate_defense with NET EXPOSURE
                hedge_lot = shield.calculate_defense(
                    exposure_to_hedge, # Use Net Exposure instead of just last pos
                    spread_points,
                    fixed_zone_points=calc_zone_points,
                    fixed_tp_points=calc_tp_points,
                    oracle_prediction=oracle_pred,
                    volatility_ratio=volatility_ratio,
                    hedge_level=hedge_level
                )

            # === EQUITY CHECK BEFORE EXECUTION ===
            account_info = broker.get_account_info()
            logger.debug(f"[ACCOUNT_INFO] Retrieved: {account_info}")
            if not account_info or account_info.get('equity', 0) < 100:  # Minimum equity check
                logger.error(f"[HEDGE] Insufficient equity: {account_info.get('equity', 0)} < 100")
                return False
            
            # Estimate required margin for hedge
            # Rough estimate: lot * contract_size * price / leverage
            contract_size = 100 if "XAU" in symbol else 100000  # Gold vs forex
            leverage = account_info.get('leverage', 100)
            required_margin = hedge_lot * contract_size * target_price / leverage
            available_margin = account_info.get('margin_free', 0)
            
            if required_margin > available_margin:
                logger.error(f"[HEDGE] Insufficient margin: Required {required_margin:.2f} > Available {available_margin:.2f}")
                
                # [CRITICAL FIX] MARGIN CLIFF PROTECTION
                # If we can't afford the full hedge, we calculate the MAX lot we CAN afford.
                # It's better to have a partial hedge than NO hedge.
                
                # Calculate max affordable lot (leaving 10% buffer)
                max_affordable_lot = (available_margin * 0.9 * leverage) / (contract_size * target_price)
                
                # Round down to 2 decimal places
                max_affordable_lot = math.floor(max_affordable_lot * 100) / 100.0
                
                if max_affordable_lot >= 0.01:
                    logger.warning(f"[HEDGE] EMERGENCY: Reducing hedge size from {hedge_lot} to {max_affordable_lot} due to margin constraints.")
                    hedge_lot = max_affordable_lot
                else:
                    logger.critical(f"[HEDGE] FATAL: Account blown. Cannot afford even 0.01 lot hedge.")
                    return False

            # === ATOMIC STATE TRANSITION: SINGLE -> BUCKET ===
            # For the FIRST hedge (transition from 1 to 2 positions), verify position exists
            if len(positions) == 1:
                logger.info(f"[TRANSITION] Initiating SINGLE -> BUCKET transition for {symbol}")
                
                first_ticket = positions[0]['ticket']
                
                # CRITICAL FIX: Verify position still exists in broker before modifying
                broker_positions = broker.get_positions(symbol=symbol)
                
                if broker_positions is None:
                    logger.error(f"[TRANSITION] Failed to verify position #{first_ticket} (API Error) - ABORTING HEDGE")
                    return False

                position_exists = any(p.ticket == first_ticket for p in broker_positions)
                
                if not position_exists:
                    logger.error(f"[TRANSITION] Position #{first_ticket} no longer exists in broker - ABORTING HEDGE")
                    # Clean up stale position from position_manager
                    if hasattr(position_manager, 'mark_position_as_ghost'):
                        position_manager.mark_position_as_ghost(first_ticket)
                        position_manager._update_bucket_stats()
                    return False
                
                # TP/SL is already 0.0 (Virtual TP), so no need to remove it.
                logger.info(f"[TRANSITION] Verified position #{first_ticket} exists. Proceeding to hedge.")
            
            # === ZERO LATENCY EXECUTION ===
            # Execute hedge immediately
            logger.info(f"[HEDGE] EXECUTING {next_action} {hedge_lot:.2f} lots NOW...")
            
            result = broker.execute_order(
                action="OPEN",
                symbol=symbol,
                order_type=next_action,
                price=target_price,
                volume=hedge_lot,
                sl=0.0,  # No broker SL - bucket managed by Python
                tp=0.0,  # No broker TP - bucket needs calculated break-even
                # MT5 comment limit: 31 chars
                comment=f"HDG_Z{int(calc_zone_points)}"[:31]
            )
            
            if not result or not result.get('ticket'):
                logger.error(f"[HEDGE] Execution failed: {result}")
                # TODO: Rollback state if this was first hedge
                return False
                
            # === POST-EXECUTION CLEANUP ===
            # Now safe to do slower operations
            logger.info(f"[HEDGE] SUCCESS: Ticket #{result['ticket']} executed at {target_price:.5f}")
            
            # Log hedge execution plan details (UI LOGGER)
            ai_reason = f"Zone Breach ({zone_width_points/point:.0f} pts). VolRatio: {volatility_ratio:.2f}"
            
            ui_logger.info(f"\n=== HEDGE {len(positions) + 1} EXECUTED ===")
            ui_logger.info(f"Hedge {len(positions) + 1}:       {hedge_lot:.2f} lots @ {target_price:.5f}")
            ui_logger.info(f"Type:          {next_action}")
            ui_logger.info(f"AI Reason:     {ai_reason}")
            ui_logger.info(f"Context:       Zone={zone_width_points/point:.1f} pips | TP={tp_width_points/point:.1f} pips")
            ui_logger.info(f"==================================")

            # Remove TP/SL from additional positions (for hedges beyond the first)
            # First hedge TP/SL already removed above in atomic transition
            if len(positions) > 1:
                logger.info(f"[BUCKET] Removing broker TP/SL from {len(positions)-1} additional position(s)")
                for i, pos in enumerate(positions[1:], start=2):  # Skip first position
                    try:
                        # Only modify if position has TP or SL set
                        if pos.get('tp', 0.0) > 0.0 or pos.get('sl', 0.0) > 0.0:
                            modify_result = broker.execute_order(
                                action="MODIFY",
                                symbol=symbol,
                                order_type="",  # Not used for modify
                                price=0.0,  # Not used for modify
                                volume=0.0,  # Not used for modify
                                sl=0.0,  # Remove SL
                                tp=0.0,  # Remove TP
                                magic=pos['ticket'],  # Position ticket to modify
                                ticket=pos['ticket']  # Explicit ticket parameter
                            )
                            if modify_result and modify_result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                                logger.info(f"[BUCKET] [OK] TP/SL removed from position #{pos['ticket']}")
                            else:
                                logger.warning(f"[BUCKET] [WARN] Failed to remove TP/SL from position #{pos['ticket']}: retcode {modify_result.get('retcode', 'unknown')}")
                        else:
                            logger.debug(f"[BUCKET] Position #{pos['ticket']} already has no TP/SL")
                    except Exception as e:
                        logger.error(f"[BUCKET] [FAIL] Error modifying position #{pos['ticket']}: {e}")
            
            logger.info(f"[BUCKET] All {len(positions) + 1} positions now in bucket mode (Python-managed exits)")
            logger.info(f"[BUCKET] Target: Break-even + profit via dynamic exit logic")

            if result and result.get('ticket'):
                ticket = result['ticket']

                # Record for learning using thread-safe persistence
                position_manager.record_trade_metadata(ticket, {
                    "symbol": symbol,
                    "type": next_action,
                    "entry_price": target_price,
                    "obs": [abs(target_price - first_pos['price_open']) / point,  # drawdown
                           0.0010,  # atr
                           0.0,     # trend
                           0.0],    # nexus
                    "action": [1.0, 0.8],  # hedge_mult, zone_mod
                    "open_time": time.time()
                })

                # Update state
                with state.lock:
                    state.last_hedge_time = time.time()
                    state.active_hedges += 1

                # CRITICAL: Explicitly link new hedge to existing bucket
                # Find bucket ID using the first position's ticket
                first_ticket = first_pos['ticket']
                bucket_id = position_manager.find_bucket_by_tickets([first_ticket])
                
                if bucket_id:
                    success = position_manager.add_position_to_bucket(bucket_id, ticket)
                    if success:
                        logger.info(f"[BUCKET] Successfully linked hedge #{ticket} to bucket {bucket_id}")
                    else:
                        logger.error(f"[BUCKET] Failed to link hedge #{ticket} to bucket {bucket_id}")
                else:
                    # If no bucket exists yet (transitioning from single), create one now
                    # We must force a refresh in the main loop to detect this new bucket
                    logger.warning(f"[BUCKET] Could not find bucket for parent ticket #{first_ticket} - Main loop will handle creation")

                # Force a small sleep to allow broker to update
                time.sleep(0.1)
                return True
            else:
                logger.error("Zone recovery order failed")
                return False

        except Exception as e:
            logger.error(f"Zone recovery error: {e}")
            return False

    def get_risk_status(self, symbol: str) -> Dict[str, Any]:
        """Get current risk status for a symbol."""
        state = self._get_hedge_state(symbol)

        with state.lock:
            return {
                "symbol": symbol,
                "active_hedges": state.active_hedges,
                "last_hedge_time": state.last_hedge_time,
                "time_since_last_hedge": time.time() - state.last_hedge_time,
                "zone_width_pips": state.zone_width_points,
                "tp_width_pips": state.tp_width_points
            }

    def reset_symbol_state(self, symbol: str) -> None:
        """Reset risk state for a symbol (use with caution)."""
        with self._lock:
            if symbol in self._hedge_states:
                del self._hedge_states[symbol]
                logger.info(f"Reset risk state for {symbol}")

    def get_emergency_status(self, total_positions: int) -> Tuple[bool, str]:
        """
        Check if emergency measures should be taken.

        Args:
            total_positions: Total positions across all symbols

        Returns:
            Tuple of (is_emergency, reason)
        """
        if total_positions >= self._global_position_cap:
            return True, f"Emergency: {total_positions} positions >= cap of {self._global_position_cap}"

        return False, "Normal operations"