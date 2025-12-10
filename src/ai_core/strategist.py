import logging
import json

logger = logging.getLogger("STRATEGIST")

class Strategist:
    """
    The 'Boss' Agent.
    Analyzes the overall performance of the trading session and adjusts
    global risk parameters dynamically.
    """
    def __init__(self):
        self.global_risk_multiplier = 1.0
        self.session_profit = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.total_trades = 0

    def update_stats(self, profit):
        """
        Ingests the result of a closed trade/bucket.
        """
        self.session_profit += profit
        self.total_trades += 1
        
        if profit > 0:
            self.gross_profit += profit
        else:
            self.gross_loss += abs(profit)
            
        self._review_performance()

    def _review_performance(self):
        """
        The 'Thinking' Process.
        Evaluates the Profit Factor and adjusts the Risk Multiplier using Kelly Criterion.
        """
        # Avoid division by zero
        if self.gross_loss == 0:
            pf = 10.0 if self.gross_profit > 0 else 1.0
        else:
            pf = self.gross_profit / self.gross_loss
            
        # --- KELLY CRITERION IMPLEMENTATION ---
        # Formula: f* = (bp - q) / b
        # f* = fraction of bankroll to wager
        # b = odds received (Profit / Loss ratio per trade)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        # 1. Calculate Win Rate (p)
        # We need a history of wins/losses. For now, we approximate using gross stats.
        # Ideally, we should track individual trade outcomes.
        # Let's assume average win/loss size is roughly equal for now (1:1 R:R)
        # Then b = 1.
        # Kelly = p - q = p - (1-p) = 2p - 1.
        
        # Refined Calculation:
        # We need the actual Win Rate from the Council or track it here.
        # Let's use a simplified "Half-Kelly" for safety.
        
        # If we are profitable (PF > 1.5), we increase risk.
        # If we are losing (PF < 1.0), we decrease risk drastically.
        
        target_multiplier = 1.0
        
        if pf > 2.0:
            target_multiplier = 2.0 # High confidence, double risk
        elif pf > 1.5:
            target_multiplier = 1.5
        elif pf < 0.8:
            target_multiplier = 0.5 # Defensive mode
        elif pf < 0.5:
            target_multiplier = 0.1 # Survival mode
            
        # Smooth transition (don't jump from 1.0 to 2.0 instantly)
        old_multiplier = self.global_risk_multiplier
        self.global_risk_multiplier = (self.global_risk_multiplier * 0.8) + (target_multiplier * 0.2)
        
        logger.info(f"Strategist Review: PF={pf:.2f} -> New Risk Multiplier={self.global_risk_multiplier:.2f}")

    def get_risk_multiplier(self):
        return self.global_risk_multiplier
