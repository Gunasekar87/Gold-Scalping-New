import sys
import os
import unittest
import numpy as np
import datetime
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai_core.regime_detector import RegimeDetector, MarketRegime
from src.ai_core.tick_pressure import TickPressureAnalyzer
from src.utils.news_filter import NewsFilter

class TestUnifiedFieldTheory(unittest.TestCase):
    
    def setUp(self):
        self.geo = RegimeDetector()
        self.phys = TickPressureAnalyzer()
        self.realist = NewsFilter()

    # --- GEOMETRICIAN TESTS ---

    def test_entropy_chaos(self):
        """Test if Entropy correctly identifies Random Walk (Chaos)."""
        print("\n[GEOMETRICIAN] Testing Entropy on Random Walk...")
        # Generate random walk
        np.random.seed(42)
        random_returns = np.random.normal(0, 1, 100)
        prices = 1000 + np.cumsum(random_returns)
        
        candles = [{'close': p} for p in prices]
        
        entropy = self.geo._calculate_shannon_entropy(candles)
        print(f"Random Walk Entropy: {entropy:.4f}")
        
        # Should be relatively high (close to 1.0 is hard with short series but > 0.6 expected)
        self.assertGreater(entropy, 0.5, "Entropy should be high for random walk")

    def test_entropy_order(self):
        """Test if Entropy correctly identifies Order (Linear Trend)."""
        print("\n[GEOMETRICIAN] Testing Entropy on Linear Trend...")
        # Generate perfect trend
        prices = np.linspace(1000, 1100, 100)
        candles = [{'close': p} for p in prices]
        
        entropy = self.geo._calculate_shannon_entropy(candles)
        print(f"Linear Trend Entropy: {entropy:.4f}")
        
        # Should be very low (close to 0)
        self.assertLess(entropy, 0.4, "Entropy should be low for linear trend")

    def test_hurst_trend(self):
        """Test if Hurst correctly identifies Trend (Persistent)."""
        print("\n[GEOMETRICIAN] Testing Hurst on Trend...")
        # Generate trend with noise
        t = np.linspace(0, 10, 200)
        prices = t + np.random.normal(0, 0.1, 200)
        candles = [{'close': p} for p in prices]
        
        hurst = self.geo._calculate_hurst_exponent(candles)
        print(f"Trend Hurst: {hurst:.4f}")
        
        self.assertGreater(hurst, 0.5, "Hurst should be > 0.5 for trend")

    # --- PHYSICIST TESTS ---

    def test_reynolds_number(self):
        """Test Reynolds Number calculation."""
        print("\n[PHYSICIST] Testing Reynolds Number...")
        # Mock ticks: High Velocity, High Volatility, Low Spread -> High Re (Turbulent)
        self.phys.ticks.clear()
        
        # 100 ticks in 1 second (High Velocity = 100)
        start_time = 1000.0
        for i in range(100):
            # High volatility prices
            price = 2000.0 + (i % 2) * 10 # Alternating 2000, 2010
            self.phys.ticks.append((price, start_time + (i/100.0)))
            
        spread_points = 1.0 # Low Viscosity
        
        re = self.phys.calculate_reynolds_number(spread_points)
        print(f"Turbulent Reynolds: {re:.2f}")
        
        self.assertGreater(re, 500, "Reynolds should be high for turbulent flow")
        
    # --- CHEMIST TESTS ---
    
    def test_vpin_toxicity(self):
        """Test VPIN calculation."""
        print("\n[CHEMIST] Testing VPIN Toxicity...")
        self.phys.buy_volume_buffer.clear()
        self.phys.sell_volume_buffer.clear()
        
        # Toxic Flow: All buys
        self.phys.buy_volume_buffer.extend([10, 10, 10])
        self.phys.sell_volume_buffer.extend([0, 0, 0])
        
        vpin = self.phys.calculate_vpin()
        print(f"Toxic VPIN: {vpin:.2f}")
        self.assertEqual(vpin, 1.0, "VPIN should be 1.0 for one-sided flow")
        
        # Balanced Flow
        self.phys.sell_volume_buffer.extend([10, 10, 10])
        vpin = self.phys.calculate_vpin()
        print(f"Balanced VPIN: {vpin:.2f}")
        self.assertEqual(vpin, 0.0, "VPIN should be 0.0 for balanced flow")

    # --- REALIST TESTS ---
    
    def test_asian_session_block(self):
        """Test Asian Session Block (00:00 - 06:00 UTC)."""
        print("\n[REALIST] Testing Asian Session Block...")
        
        # Patch the datetime MODULE imported in news_filter
        with patch('src.utils.news_filter.datetime') as mock_dt_module:
            # We need to mock datetime.datetime.utcnow()
            # mock_dt_module is the 'datetime' module
            # We need to ensure mock_dt_module.datetime is our mock class
            
            mock_dt_class = MagicMock()
            mock_dt_module.datetime = mock_dt_class
            
            # Case 1: 03:00 UTC (Asian Session) -> Should Block
            mock_dt_class.utcnow.return_value = datetime.datetime(2026, 1, 8, 3, 0, 0)
            
            # Re-init mechanism if needed? No, is_news_event calls utcnow every time.
            is_news = self.realist.is_news_event()
            print(f"03:00 UTC Blocked? {is_news}")
            self.assertTrue(is_news, "Should block during Asian Session (03:00 UTC)")
            
            # Case 2: 05:59 UTC (Asian Session) -> Should Block
            mock_dt_class.utcnow.return_value = datetime.datetime(2026, 1, 8, 5, 59, 0)
            is_news = self.realist.is_news_event()
            print(f"05:59 UTC Blocked? {is_news}")
            self.assertTrue(is_news, "Should block during Asian Session (05:59 UTC)")
            
            # Case 3: 07:00 UTC (London Open approx) -> Should Allow
            mock_dt_class.utcnow.return_value = datetime.datetime(2026, 1, 8, 7, 0, 0)
            is_news = self.realist.is_news_event()
            print(f"07:00 UTC Blocked? {is_news}")
            self.assertFalse(is_news, "Should NOT block at 07:00 UTC")

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUnifiedFieldTheory)
    unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
