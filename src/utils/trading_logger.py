"""
Trading Logger - Structured logging for clean, actionable trade information
Reduces log noise by only showing:
1. Initial trade entries with full trading plan
2. Trade exits with detailed summaries
3. Decision changes (not repetitive HOLD signals)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

# Use a specific logger for UI output that will be configured to show on console
logger = logging.getLogger("AETHER_UI")


class TradingLogger:
    """Structured logger for trading operations with minimal noise"""
    
    @staticmethod
    def log_initial_trade(bucket_id: str, data: Dict):
        """
        Log comprehensive trading plan when opening initial trade
        
        Args:
            bucket_id: Unique bucket identifier
            data: Trade data containing:
                - action: BUY/SELL
                - symbol: Trading symbol
                - lots: Position size
                - entry_price: Entry price
                - tp_price: Take profit target
                - tp_atr: ATR multiplier for TP
                - tp_pips: TP distance in pips
                - hedges: List of hedge levels (projected)
                - atr_pips: Current ATR in pips
                - reasoning: Detailed AI reasoning
        """
        # Format the output exactly as requested
        msg = []
        msg.append("TRADE ENTRY SUMMARY")
        msg.append("-" * 45)
        msg.append(f"Symbol: {data['symbol']}")
        msg.append(f"Action: {data['action']}")
        msg.append(f"Entry Price: {data['entry_price']:.5f}")
        msg.append(f"Position Size: {data['lots']} lots")
        msg.append(f"Target: {data['tp_price']:.5f} (+{data['tp_pips']:.1f} pips)")
        
        # Display hedge levels (projected)
        if 'hedges' in data and data['hedges']:
            msg.append("Hedge Plan:")
            for i, hedge in enumerate(data['hedges'], 1):
                msg.append(f"  Hedge {i}: {hedge['direction']} {hedge['lots']} lots @ {hedge['trigger_price']}")
        
        msg.append(f"AI Analysis: {data['reasoning']}")
        
        logger.info("\n".join(msg))
    
    @staticmethod
    def log_trade_close(symbol: str, exit_reason: str, duration: float, pnl_usd: float, pnl_pips: float, positions_closed: int, ai_analysis: Dict):
        """
        Log trade exit with detailed performance summary
        """
        # Format duration
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)
        duration_str = f"{int(h)}h {int(m)}m {int(s)}s" if h > 0 else f"{int(m)}m {int(s)}s"
        
        # Construct explanation if not provided
        explanation = ai_analysis.get('explanation', 'N/A')
        
        msg = []
        msg.append("TRADE EXIT SUMMARY")
        msg.append("-" * 45)
        msg.append(f"Symbol: {symbol}")
        msg.append(f"Exit Reason: {exit_reason}")
        msg.append(f"Duration: {duration_str}")
        msg.append(f"Total PnL: ${pnl_usd:.2f} ({'+' if pnl_pips >= 0 else ''}{pnl_pips:.1f} pips)")
        msg.append(f"Positions Closed: {positions_closed}")
        msg.append(f"Explanation: {explanation}")
        
        logger.info("\n".join(msg))
    
    @staticmethod
    def log_decision_change(symbol: str, old_decision: Dict, new_decision: Dict, reason: str):
        """
        Log when AI decision changes significantly
        
        Args:
            symbol: Trading symbol
            old_decision: Previous decision state
            new_decision: Current decision state
            reason: What changed and why
        """
        # This might be too noisy for the main terminal based on user request.
        # We'll log it to the file logger (standard logging) instead of UI logger.
        file_logger = logging.getLogger("TradingLogger")
        file_logger.info(f"[DECISION] {symbol} changed from {old_decision.get('action')} to {new_decision.get('action')} | {reason}")
    
    @staticmethod
    def log_error(context: str, error: Exception, additional_info: Optional[Dict] = None):
        """
        Log errors with context
        
        Args:
            context: What was being attempted
            error: Exception that occurred
            additional_info: Optional additional context
        """
        print(f"\n[ERROR] {context}")
        print(f"  Error: {str(error)}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"  {key}: {value}")
        print()
    
    @staticmethod
    def log_system_event(event_type: str, message: str, data: Optional[Dict] = None):
        """
        Log system-level events (startup, shutdown, connection, etc.)
        
        Args:
            event_type: Type of event (STARTUP, SHUTDOWN, CONNECTION, etc.)
            message: Event description
            data: Optional event data
        """
        print(f"\n[{event_type}] {message}")
        if data:
            for key, value in data.items():
                print(f"  {key}: {value}")
        print()


class DecisionTracker:
    """
    Tracks AI decisions to detect meaningful changes
    Prevents repetitive HOLD signal logging
    """
    
    def __init__(self):
        self._last_decisions = {}  # symbol -> last_decision
        self._decision_history = {}  # symbol -> list of recent decisions
        
    def should_log_decision(self, symbol: str, current_decision: Dict) -> tuple[bool, Optional[str]]:
        """
        Determine if current decision should be logged
        
        Args:
            symbol: Trading symbol
            current_decision: Current AI decision with keys:
                - action: BUY/SELL/HOLD
                - confidence: Float 0-1
                - reasoning: String explanation
        
        Returns:
            Tuple of (should_log, change_reason)
        """
        last_decision = self._last_decisions.get(symbol)
        
        if not last_decision:
            # First decision for this symbol - log it
            self._update_decision(symbol, current_decision)
            return True, "Initial signal"
        
        # Check for significant changes
        action_changed = last_decision['action'] != current_decision['action']
        confidence_changed = abs(last_decision['confidence'] - current_decision['confidence']) > 0.15
        
        if action_changed:
            reason = f"Action changed: {last_decision['action']} -> {current_decision['action']}"
            self._update_decision(symbol, current_decision)
            return True, reason
        
        if confidence_changed:
            reason = f"Confidence shifted: {last_decision['confidence']:.3f} -> {current_decision['confidence']:.3f}"
            self._update_decision(symbol, current_decision)
            return True, reason
        
        # No significant change - don't log repetitive signals
        return False, None
    
    def _update_decision(self, symbol: str, decision: Dict):
        """Update stored decision for symbol"""
        self._last_decisions[symbol] = {
            'action': decision['action'],
            'confidence': decision['confidence'],
            'timestamp': datetime.now(),
            'reasoning': decision.get('reasoning', '')
        }
        
        # Maintain history (last 10 decisions)
        if symbol not in self._decision_history:
            self._decision_history[symbol] = []
        
        self._decision_history[symbol].append(decision.copy())
        
        # Keep only recent history
        if len(self._decision_history[symbol]) > 10:
            self._decision_history[symbol] = self._decision_history[symbol][-10:]
    
    def get_decision_summary(self, symbol: str) -> Optional[Dict]:
        """Get summary of recent decisions for a symbol"""
        if symbol not in self._decision_history:
            return None
        
        history = self._decision_history[symbol]
        
        return {
            'total_decisions': len(history),
            'last_action': history[-1]['action'] if history else None,
            'avg_confidence': sum(d['confidence'] for d in history) / len(history) if history else 0,
            'action_distribution': {
                'BUY': sum(1 for d in history if d['action'] == 'BUY'),
                'SELL': sum(1 for d in history if d['action'] == 'SELL'),
                'HOLD': sum(1 for d in history if d['action'] == 'HOLD')
            }
        }


def format_pips(value: float, symbol: str) -> str:
    """
    Format pip value correctly based on symbol type
    
    Args:
        value: Pip value to format
        symbol: Trading symbol (e.g., EURUSD, XAUUSD, USDJPY)
    
    Returns:
        Formatted string with correct decimal places
    """
    # JPY pairs and gold use 2 decimal places (100x multiplier)
    if 'JPY' in symbol or 'XAU' in symbol or 'XAG' in symbol:
        return f"{value:.2f}"
    # Major forex pairs use 1 decimal place (10000x multiplier)
    else:
        return f"{value:.1f}"
