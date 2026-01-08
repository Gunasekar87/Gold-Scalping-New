
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_manager import RiskManager, ZoneConfig

class TestHedgeIntelligence(unittest.TestCase):
    def setUp(self):
        config = ZoneConfig(zone_pips=20.0, tp_pips=25.0)
        self.rm = RiskManager(config)
        # Mock dependencies
        self.broker = MagicMock()
        self.broker.is_trade_allowed.return_value = True
        self.shield = MagicMock()
        self.ppo = MagicMock()
        self.pm = MagicMock()
        
        # Default mocks
        self.rm.calculate_zone_parameters = MagicMock(return_value=(200, 200)) # 20 pips
        self.rm._get_hedge_state = MagicMock()
        self.rm.validate_hedge_conditions = MagicMock(return_value=(True, "OK"))
        
        # Mock dependencies for _get_hedge_state
        mock_state = MagicMock()
        mock_state.lock = MagicMock()
        mock_state.lock.__enter__ = MagicMock()
        mock_state.lock.__exit__ = MagicMock()
        self.rm._get_hedge_state.return_value = mock_state

    @patch('src.utils.hedge_coordinator.get_hedge_coordinator')
    @patch('src.ai_core.multi_horizon_predictor.get_multi_horizon_predictor')
    @patch('src.ai_core.smart_hedge_timing.get_smart_hedge_timing')
    def test_vpin_block(self, mock_get_smart, mock_get_multi, mock_get_coord):
        """Test if Toxic Flow (VPIN > 0.6) blocks hedging."""
        print("\n[TEST] VPIN Toxicity Block")
        
        # Setup mocks to allow execution to proceed
        mock_coord = MagicMock()
        mock_coord.can_hedge_bucket.return_value = (True, "OK")
        mock_get_coord.return_value = mock_coord
        
        mock_smart = MagicMock()
        # Ensure smart timing doesn't block
        mock_decision = MagicMock()
        mock_decision.should_hedge = True
        mock_decision.timing = 'NOW'
        mock_decision.size_multiplier = 1.0
        mock_smart.should_hedge_now.return_value = mock_decision
        mock_get_smart.return_value = mock_smart
        
        # Setup toxic pressure
        pressure = {
            'chemistry': {'vpin': 0.8}, # Toxic
            'physics': {'reynolds_number': 100}
        }
        
        # Execute
        result = self.rm.execute_zone_recovery(
            self.broker, "XAUUSD", [{'ticket': 1, 'time': 1000}], 
            {'tick_age_s': 0.1, 'time': 1234567890.0, 'candle_close_age_s': 1.0}, # FRESH DATA
            0.01,
            self.shield, self.ppo, self.pm, strict_entry=False,
            pressure_metrics=pressure
        )
        
        self.assertFalse(result, "Hedge should be BLOCKED due to High VPIN")
        print("✅ VPIN > 0.6 Blocked Hedge successfully")

    @patch('src.utils.hedge_coordinator.get_hedge_coordinator')
    @patch('src.ai_core.multi_horizon_predictor.get_multi_horizon_predictor')
    @patch('src.ai_core.smart_hedge_timing.get_smart_hedge_timing')
    def test_reynolds_widen(self, mock_get_smart, mock_get_multi, mock_get_coord):
        """Test if High Reynolds Number widens the zone."""
        print("\n[TEST] Reynolds Turbulence Widen")
        
        # Setup mocks
        mock_coord = MagicMock()
        mock_coord.can_hedge_bucket.return_value = (True, "OK")
        mock_get_coord.return_value = mock_coord
        
        mock_smart = MagicMock()
        mock_decision = MagicMock()
        mock_decision.should_hedge = True
        mock_decision.timing = 'NOW'
        mock_decision.size_multiplier = 1.0
        mock_smart.should_hedge_now.return_value = mock_decision
        mock_get_smart.return_value = mock_smart
        
        # Setup turbulent pressure
        pressure = {
            'chemistry': {'vpin': 0.1},
            'physics': {'reynolds_number': 2500} # Turbulent
        }
        
        # We need atr_val to avoid early exit
        with patch('src.risk_manager.logger') as mock_logger:
            result = self.rm.execute_zone_recovery(
                self.broker, "XAUUSD", [{'ticket': 1, 'time': 1000, 'type': 0, 'price_open': 2000.0}], 
                {'bid': 1999.0, 'ask': 1999.5, 'tick_age_s': 0.1, 'time': 1234567890.0, 'candle_close_age_s': 1.0}, # FRESH DATA
                0.01,
                self.shield, self.ppo, self.pm, strict_entry=False,
                atr_val=1.0, 
                pressure_metrics=pressure
            )
            
            # Check for log message containing "Widening Zone"
            log_found = False
            print(f"Result of execution: {result}")
            print("Captured Logs:")
            for call in mock_logger.info.call_args_list:
                if not call[0]: continue
                msg = str(call[0][0])
                print(f"LOG: {msg}")
                if "Widening Zone" in msg:
                    log_found = True
                    print(f"✅ Found Log: {msg}")
                    break
            
            self.assertTrue(log_found, "Should log 'Widening Zone' when Reynolds > 2000")

if __name__ == '__main__':
    unittest.main()
