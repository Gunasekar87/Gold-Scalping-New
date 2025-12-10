import logging
import random
import time
import yfinance as yf

logger = logging.getLogger("SentimentEngine")

class SentimentEngine:
    """
    The 'News Reader' Agent.
    Uses Yahoo Finance (yfinance) to fetch real-time news headlines.
    """
    def __init__(self):
        self.last_update = 0
        self.current_sentiment = 0.0 # -1.0 (Bearish) to 1.0 (Bullish)
        
        # Symbol Mapping (MT5 -> Yahoo Finance)
        self.ticker_map = {
            "XAUUSD": "XAUUSD=X", # Gold Spot
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD",
            "US500": "^GSPC", # S&P 500
            "USTEC": "^IXIC"  # Nasdaq
        }
        
        # Institutional Financial Lexicon (Enhanced with Context)
        # We use a simple "N-Gram" approach to detect negations.
        self.bullish_words = {
            "surge", "jump", "soar", "record", "high", "gain", "bull", "optimism", 
            "growth", "beat", "exceed", "positive", "upgrade", "buy", "rally", "rocket", "breakout"
        }
        self.bearish_words = {
            "plunge", "drop", "fall", "crash", "low", "loss", "bear", "pessimism", 
            "recession", "miss", "negative", "downgrade", "sell", "slump", "fear", "panic", "collapse"
        }
        self.negations = {"not", "no", "never", "despite", "although", "barely"}

    def fetch_news(self, symbol):
        """
        Fetches the latest headlines for a symbol using Yahoo Finance.
        """
        yahoo_symbol = self.ticker_map.get(symbol, symbol)
        try:
            ticker = yf.Ticker(yahoo_symbol)
            news = ticker.news
            headlines = [item['title'] for item in news if 'title' in item]
            return headlines
        except Exception as e:
            # logger.warning(f"Failed to fetch news for {symbol} ({yahoo_symbol}): {e}")
            return []

    def analyze(self, symbol):
        """
        Returns the current sentiment score using Context-Aware NLP.
        """
        # Debounce: Update every 60 seconds max
        if time.time() - self.last_update < 60:
            return self.current_sentiment
            
        headlines = self.fetch_news(symbol)
        if not headlines:
            # Decay sentiment towards neutral over time
            self.current_sentiment *= 0.95
            # Prevent API spam by updating timestamp even on failure
            self.last_update = time.time()
            return self.current_sentiment
            
        score = 0.0
        count = 0
        
        for headline in headlines:
            logger.info(f"NEWS: {headline}")
            words = headline.lower().split()
            h_score = 0
            
            # Context Window (Look back 1 word for negation)
            for i, word in enumerate(words):
                multiplier = 1
                if i > 0 and words[i-1] in self.negations:
                    multiplier = -1 # Invert sentiment
                
                if word in self.bullish_words: 
                    h_score += (1 * multiplier)
                elif word in self.bearish_words: 
                    h_score -= (1 * multiplier)
            
            # Normalize headline score
            if h_score > 0: score += 1.0
            elif h_score < 0: score -= 1.0
            count += 1
            
        if count > 0:
            # Update Moving Average of Sentiment
            new_score = score / count
            # Smooth update (20% weight to new news)
            self.current_sentiment = (self.current_sentiment * 0.8) + (new_score * 0.2)
            self.last_update = time.time()
            
        return self.current_sentiment
