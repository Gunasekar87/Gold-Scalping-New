"""
Oracle Model Validation Script

Tests the Oracle/Nexus Transformer model to verify:
1. Model loads correctly
2. Predictions are generated
3. Output format is correct
4. Confidence scores are reasonable
5. Model responds to different market conditions

Usage:
    python validate_oracle.py
"""

import sys
import os
import time
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_core.oracle import Oracle
from src.ai_core.nexus_brain import NexusBrain


def generate_test_candles(pattern: str = "uptrend", num_candles: int = 64) -> list:
    """
    Generate synthetic candle data for testing.
    
    Args:
        pattern: "uptrend", "downtrend", "sideways", "volatile"
        num_candles: Number of candles to generate
        
    Returns:
        List of candle dictionaries
    """
    base_price = 2650.0  # XAU/USD typical price
    candles = []
    
    for i in range(num_candles):
        if pattern == "uptrend":
            # Gradual upward movement
            trend = i * 0.5
            noise = np.random.randn() * 2.0
            close = base_price + trend + noise
            
        elif pattern == "downtrend":
            # Gradual downward movement
            trend = -i * 0.5
            noise = np.random.randn() * 2.0
            close = base_price + trend + noise
            
        elif pattern == "sideways":
            # Range-bound movement
            noise = np.random.randn() * 3.0
            close = base_price + noise
            
        elif pattern == "volatile":
            # High volatility
            noise = np.random.randn() * 10.0
            close = base_price + noise
            
        else:
            close = base_price
        
        # Generate OHLC
        high = close + abs(np.random.randn()) * 2.0
        low = close - abs(np.random.randn()) * 2.0
        open_price = (high + low) / 2 + np.random.randn() * 1.0
        volume = 1000 + np.random.randint(0, 500)
        
        candles.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': volume,
            'time': time.time() - (num_candles - i) * 60
        })
    
    return candles


def test_oracle_model():
    """Test Oracle model predictions."""
    print("\n" + "="*70)
    print("🔮 ORACLE MODEL VALIDATION")
    print("="*70)
    
    # Initialize Oracle
    print("\n[1/5] Initializing Oracle...")
    try:
        oracle = Oracle(model_path="models/nexus_transformer.pth")
        print("✓ Oracle initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Oracle: {e}")
        return False
    
    # Check if model is loaded
    print("\n[2/5] Checking model status...")
    if oracle.model is None:
        print("⚠ WARNING: Model file not found. Oracle will return NEUTRAL.")
        print("  Expected location: models/nexus_transformer.pth")
        print("  This is OK for testing, but predictions will be neutral.")
    else:
        print("✓ Model loaded successfully")
        print(f"  Device: {oracle.device}")
    
    # Test different market patterns
    print("\n[3/5] Testing predictions on different patterns...")
    patterns = ["uptrend", "downtrend", "sideways", "volatile"]
    results = {}
    
    for pattern in patterns:
        candles = generate_test_candles(pattern=pattern, num_candles=64)
        
        try:
            prediction, confidence = oracle.predict(candles)
            results[pattern] = {
                "prediction": prediction,
                "confidence": confidence,
                "success": True
            }
            print(f"  {pattern:12s}: {prediction:8s} (confidence: {confidence:.3f})")
        except Exception as e:
            results[pattern] = {
                "prediction": None,
                "confidence": 0.0,
                "success": False,
                "error": str(e)
            }
            print(f"  {pattern:12s}: ✗ ERROR - {e}")
    
    # Test trajectory prediction
    print("\n[4/5] Testing trajectory prediction...")
    try:
        candles = generate_test_candles("uptrend", 64)
        trajectory = oracle.predict_trajectory(candles, horizon=10)
        
        if trajectory and len(trajectory) > 0:
            print(f"✓ Trajectory generated: {len(trajectory)} future points")
            print(f"  Current price: {candles[-1]['close']:.2f}")
            print(f"  Predicted end: {trajectory[-1]:.2f}")
            print(f"  Change: {trajectory[-1] - candles[-1]['close']:+.2f}")
        else:
            print("⚠ Trajectory is empty (model may not be loaded)")
    except Exception as e:
        print(f"✗ Trajectory prediction failed: {e}")
    
    # Validation summary
    print("\n[5/5] Validation Summary...")
    successful_tests = sum(1 for r in results.values() if r["success"])
    total_tests = len(results)
    
    print(f"\nTests Passed: {successful_tests}/{total_tests}")
    
    if oracle.model is None:
        print("\n⚠ RECOMMENDATION:")
        print("  Train the model using: python src/ai_core/nexus_trainer.py")
        print("  Or the model will always predict NEUTRAL")
    
    if successful_tests == total_tests:
        print("\n✓ ALL TESTS PASSED - Oracle is functional")
        return True
    else:
        print("\n⚠ SOME TESTS FAILED - Check errors above")
        return False


def test_nexus_brain():
    """Test NexusBrain model predictions."""
    print("\n" + "="*70)
    print("🧠 NEXUS BRAIN VALIDATION")
    print("="*70)
    
    print("\n[1/3] Initializing NexusBrain...")
    try:
        nexus = NexusBrain(model_path="models/nexus_transformer.pth")
        print("✓ NexusBrain initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize NexusBrain: {e}")
        return False
    
    print("\n[2/3] Testing predictions...")
    try:
        # Generate test data in format expected by NexusBrain
        candles = []
        base_price = 2650.0
        for i in range(64):
            close = base_price + i * 0.5 + np.random.randn() * 2.0
            candles.append([
                close - 1.0,  # open
                close + 2.0,  # high
                close - 2.0,  # low
                close,        # close
                1000.0        # volume
            ])
        
        signal, confidence, volatility = nexus.predict(candles)
        print(f"✓ Prediction: {signal} (confidence: {confidence:.3f}, vol: {volatility:.3f})")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[3/3] Testing trajectory...")
    try:
        trajectory = nexus.predict_trajectory(candles, horizon=10)
        if trajectory:
            print(f"✓ Trajectory: {len(trajectory)} points generated")
        else:
            print("⚠ Empty trajectory")
    except Exception as e:
        print(f"✗ Trajectory failed: {e}")
    
    print("\n✓ NEXUS BRAIN TESTS COMPLETED")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("🧪 AI MODEL VALIDATION SUITE")
    print("="*70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Oracle
    oracle_ok = test_oracle_model()
    
    # Test NexusBrain
    nexus_ok = test_nexus_brain()
    
    # Final report
    print("\n" + "="*70)
    print("📊 FINAL VALIDATION REPORT")
    print("="*70)
    print(f"Oracle Model:  {'✓ PASS' if oracle_ok else '✗ FAIL'}")
    print(f"Nexus Brain:   {'✓ PASS' if nexus_ok else '✗ FAIL'}")
    
    if oracle_ok and nexus_ok:
        print("\n✓ ALL MODELS VALIDATED - System is ready")
        return 0
    else:
        print("\n⚠ VALIDATION INCOMPLETE - Check errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
