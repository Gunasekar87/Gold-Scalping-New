import datetime
import logging

logger = logging.getLogger("NewsFilter")

class NewsFilter:
    """
    Simple Calendar Filter to block trading during high-impact news events.
    """
    def __init__(self):
        self.blocked_times = []
        # Example: Block NFP (First Friday of month, 8:30 AM EST)
        # This is a simplified implementation.
        
    def is_news_event(self) -> bool:
        """
        Checks if currently in a high-impact news window.
        """
        now = datetime.datetime.now()
        
        # 1. BLOCK WEEKENDS (Friday 22:00 to Sunday 22:00)
        if now.weekday() == 4 and now.hour >= 22: # Friday night
            return True
        if now.weekday() == 5: # Saturday
            return True
        if now.weekday() == 6 and now.hour < 22: # Sunday morning
            return True
            
        # 2. BLOCK SPECIFIC HIGH VOLATILITY WINDOWS (UTC)
        # Example: US Market Open Volatility (13:30 - 14:00 UTC)
        current_time_utc = datetime.datetime.utcnow().time()
        
        # Convert to simple float for comparison (e.g. 13.5 = 13:30)
        current_hour = current_time_utc.hour + (current_time_utc.minute / 60.0)
        
        # Block 13:25 - 13:35 UTC (US Open)
        if 13.41 <= current_hour <= 13.58:
            # logger.info("📰 NEWS: US Market Open Volatility Window")
            return True
            
        return False

    def should_trade(self) -> bool:
        """
        Returns True if safe to trade, False if news event.
        """
        if self.is_news_event():
            return False
        return True
