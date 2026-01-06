"""
Backtesting Framework for AETHER Trading System

Tests the trading strategy on historical data to evaluate:
- Win rate
- Profit factor
- Maximum drawdown
- Sharpe ratio
- Trade distribution

Usage:
    python backtest.py --symbol XAUUSD --start 2025-12-01 --end 2026-01-01
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.ai_core.oracle import Oracle
from src.ai_core.nexus_brain import NexusBrain


@dataclass
class Trade:
    """Single trade record."""
    entry_time: float
    entry_price: float
    direction: str  # "BUY" or "SELL"
    lot_size: float
    confidence: float
    
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    profit_pips: Optional[float] = None
    profit_usd: Optional[float] = None
    duration_minutes: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestResults:
    """Backtest performance metrics."""
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    total_profit_usd: float
    total_profit_pips: float
    avg_win_usd: float
    avg_loss_usd: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    
    # Trade details
    trades: List[Trade]
    equity_curve: List[float]
    
    # Timing
    start_date: str
    end_date: str
    duration_days: int


class Backtester:
    """
    Backtesting engine for trading strategies.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        pip_value: float = 1.0,  # USD per pip for 0.01 lot
        spread_pips: float = 2.0,
        tp_pips: float = 30.0,
        sl_pips: float = 50.0
    ):
        self.initial_balance = initial_balance
        self.lot_size = lot_size
        self.pip_value = pip_value
        self.spread_pips = spread_pips
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        
        # Initialize AI models
        self.oracle = Oracle(model_path="models/nexus_transformer.pth")
        self.nexus = NexusBrain(model_path="models/nexus_transformer.pth")
        
        # State
        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
        
        print(f"Backtester initialized:")
        print(f"  Initial Balance: ${initial_balance:,.2f}")
        print(f"  Lot Size: {lot_size}")
        print(f"  TP: {tp_pips} pips | SL: {sl_pips} pips")
    
    def generate_synthetic_data(
        self,
        start_date: str,
        end_date: str,
        timeframe: str = "M1"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLC data for testing.
        
        In production, this would load real historical data from MT5 or CSV.
        """
        print(f"\nGenerating synthetic data from {start_date} to {end_date}...")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Generate timestamps (M1 = 1 minute)
        minutes = int((end - start).total_seconds() / 60)
        timestamps = [start + timedelta(minutes=i) for i in range(minutes)]
        
        # Generate price data (random walk with trend)
        base_price = 2650.0
        prices = [base_price]
        
        for i in range(1, len(timestamps)):
            # Add trend + noise
            trend = 0.001 * np.sin(i / 1000)  # Cyclical trend
            noise = np.random.randn() * 0.5
            change = trend + noise
            prices.append(prices[-1] + change)
        
        # Generate OHLC
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            high = close + abs(np.random.randn()) * 0.5
            low = close - abs(np.random.randn()) * 0.5
            open_price = prices[i-1] if i > 0 else close
            volume = 1000 + np.random.randint(0, 500)
            
            data.append({
                'time': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'tick_volume': volume
            })
        
        df = pd.DataFrame(data)
        print(f"✓ Generated {len(df)} candles")
        return df
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLC data
            confidence_threshold: Minimum confidence to enter trade
            
        Returns:
            BacktestResults object
        """
        print(f"\nRunning backtest on {len(data)} candles...")
        print(f"Confidence threshold: {confidence_threshold}")
        
        for i in range(64, len(data)):  # Need 64 candles for Oracle
            current_candle = data.iloc[i]
            current_price = current_candle['close']
            current_time = current_candle['time'].timestamp()
            
            # Get last 64 candles for prediction
            candles = data.iloc[i-64:i].to_dict('records')
            
            # Check if we have an open trade
            if self.open_trade:
                self._manage_open_trade(current_candle)
            else:
                # Get AI prediction
                try:
                    prediction, confidence = self.oracle.predict(candles)
                    
                    # Enter trade if confidence is high enough
                    if confidence >= confidence_threshold and prediction != "NEUTRAL":
                        direction = "BUY" if prediction == "UP" else "SELL"
                        self._enter_trade(
                            current_time,
                            current_price,
                            direction,
                            confidence
                        )
                except Exception as e:
                    # Oracle may not be loaded, skip
                    pass
            
            # Update equity curve
            current_equity = self._calculate_equity(current_price)
            self.equity_curve.append(current_equity)
        
        # Close any remaining open trade
        if self.open_trade:
            last_candle = data.iloc[-1]
            self._close_trade(
                last_candle['time'].timestamp(),
                last_candle['close'],
                "End of backtest"
            )
        
        # Calculate results
        results = self._calculate_results(data)
        return results
    
    def _enter_trade(
        self,
        time: float,
        price: float,
        direction: str,
        confidence: float
    ) -> None:
        """Enter a new trade."""
        # Apply spread
        entry_price = price + (self.spread_pips / 100) if direction == "BUY" else price - (self.spread_pips / 100)
        
        self.open_trade = Trade(
            entry_time=time,
            entry_price=entry_price,
            direction=direction,
            lot_size=self.lot_size,
            confidence=confidence
        )
        
        print(f"  [{datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M')}] "
              f"ENTER {direction} @ {entry_price:.2f} (conf: {confidence:.2f})")
    
    def _manage_open_trade(self, candle: pd.Series) -> None:
        """Check if open trade hits TP/SL."""
        if not self.open_trade:
            return
        
        current_price = candle['close']
        high = candle['high']
        low = candle['low']
        
        if self.open_trade.direction == "BUY":
            # Check TP
            tp_price = self.open_trade.entry_price + (self.tp_pips / 100)
            if high >= tp_price:
                self._close_trade(candle['time'].timestamp(), tp_price, "TP Hit")
                return
            
            # Check SL
            sl_price = self.open_trade.entry_price - (self.sl_pips / 100)
            if low <= sl_price:
                self._close_trade(candle['time'].timestamp(), sl_price, "SL Hit")
                return
        
        else:  # SELL
            # Check TP
            tp_price = self.open_trade.entry_price - (self.tp_pips / 100)
            if low <= tp_price:
                self._close_trade(candle['time'].timestamp(), tp_price, "TP Hit")
                return
            
            # Check SL
            sl_price = self.open_trade.entry_price + (self.sl_pips / 100)
            if high >= sl_price:
                self._close_trade(candle['time'].timestamp(), sl_price, "SL Hit")
                return
    
    def _close_trade(self, time: float, price: float, reason: str) -> None:
        """Close the open trade."""
        if not self.open_trade:
            return
        
        # Calculate profit
        if self.open_trade.direction == "BUY":
            profit_pips = (price - self.open_trade.entry_price) * 100
        else:
            profit_pips = (self.open_trade.entry_price - price) * 100
        
        profit_usd = profit_pips * self.pip_value * self.open_trade.lot_size
        
        # Update trade
        self.open_trade.exit_time = time
        self.open_trade.exit_price = price
        self.open_trade.profit_pips = profit_pips
        self.open_trade.profit_usd = profit_usd
        self.open_trade.duration_minutes = (time - self.open_trade.entry_time) / 60
        self.open_trade.exit_reason = reason
        
        # Update balance
        self.balance += profit_usd
        
        # Save trade
        self.trades.append(self.open_trade)
        
        print(f"  [{datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M')}] "
              f"CLOSE {self.open_trade.direction} @ {price:.2f} | "
              f"P/L: ${profit_usd:+.2f} ({profit_pips:+.1f} pips) | {reason}")
        
        self.open_trade = None
    
    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including open trade."""
        equity = self.balance
        
        if self.open_trade:
            if self.open_trade.direction == "BUY":
                profit_pips = (current_price - self.open_trade.entry_price) * 100
            else:
                profit_pips = (self.open_trade.entry_price - current_price) * 100
            
            unrealized_profit = profit_pips * self.pip_value * self.open_trade.lot_size
            equity += unrealized_profit
        
        return equity
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResults:
        """Calculate backtest performance metrics."""
        if not self.trades:
            print("\n⚠ No trades executed during backtest")
            return BacktestResults(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_profit_usd=0.0,
                total_profit_pips=0.0,
                avg_win_usd=0.0,
                avg_loss_usd=0.0,
                profit_factor=0.0,
                max_drawdown_usd=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=self.equity_curve,
                start_date=data.iloc[0]['time'].strftime('%Y-%m-%d'),
                end_date=data.iloc[-1]['time'].strftime('%Y-%m-%d'),
                duration_days=(data.iloc[-1]['time'] - data.iloc[0]['time']).days
            )
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.profit_usd > 0]
        losing_trades = [t for t in self.trades if t.profit_usd <= 0]
        
        total_trades = len(self.trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0
        
        # Profit metrics
        total_profit_usd = sum(t.profit_usd for t in self.trades)
        total_profit_pips = sum(t.profit_pips for t in self.trades)
        
        avg_win_usd = np.mean([t.profit_usd for t in winning_trades]) if winning_trades else 0.0
        avg_loss_usd = np.mean([t.profit_usd for t in losing_trades]) if losing_trades else 0.0
        
        gross_profit = sum(t.profit_usd for t in winning_trades)
        gross_loss = abs(sum(t.profit_usd for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = running_max - equity_array
        max_drawdown_usd = np.max(drawdown)
        max_drawdown_pct = (max_drawdown_usd / self.initial_balance * 100) if self.initial_balance > 0 else 0.0
        
        # Sharpe ratio (simplified)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 0 and np.std(returns) > 0 else 0.0
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            total_profit_usd=total_profit_usd,
            total_profit_pips=total_profit_pips,
            avg_win_usd=avg_win_usd,
            avg_loss_usd=avg_loss_usd,
            profit_factor=profit_factor,
            max_drawdown_usd=max_drawdown_usd,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=self.equity_curve,
            start_date=data.iloc[0]['time'].strftime('%Y-%m-%d'),
            end_date=data.iloc[-1]['time'].strftime('%Y-%m-%d'),
            duration_days=(data.iloc[-1]['time'] - data.iloc[0]['time']).days
        )
    
    def print_results(self, results: BacktestResults) -> None:
        """Print formatted backtest results."""
        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)
        print(f"Period: {results.start_date} to {results.end_date} ({results.duration_days} days)")
        print(f"\n💰 PROFIT METRICS")
        print(f"  Total Profit:     ${results.total_profit_usd:+,.2f} ({results.total_profit_pips:+,.1f} pips)")
        print(f"  Initial Balance:  ${self.initial_balance:,.2f}")
        print(f"  Final Balance:    ${self.balance:,.2f}")
        print(f"  Return:           {(self.balance - self.initial_balance) / self.initial_balance * 100:+.2f}%")
        
        print(f"\n📈 TRADE STATISTICS")
        print(f"  Total Trades:     {results.total_trades}")
        print(f"  Winning Trades:   {results.winning_trades}")
        print(f"  Losing Trades:    {results.losing_trades}")
        print(f"  Win Rate:         {results.win_rate:.1f}%")
        print(f"  Profit Factor:    {results.profit_factor:.2f}")
        
        print(f"\n💵 AVERAGE TRADE")
        print(f"  Avg Win:          ${results.avg_win_usd:+,.2f}")
        print(f"  Avg Loss:         ${results.avg_loss_usd:+,.2f}")
        
        print(f"\n⚠️  RISK METRICS")
        print(f"  Max Drawdown:     ${results.max_drawdown_usd:,.2f} ({results.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio:     {results.sharpe_ratio:.2f}")
        
        print("="*70)
        
        # Save to file
        self._save_results(results)
    
    def _save_results(self, results: BacktestResults) -> None:
        """Save backtest results to JSON file."""
        output_dir = Path("backtest_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"backtest_{timestamp}.json"
        
        # Convert to dict (excluding trades for brevity)
        results_dict = {
            "summary": {
                "total_trades": results.total_trades,
                "win_rate": results.win_rate,
                "total_profit_usd": results.total_profit_usd,
                "profit_factor": results.profit_factor,
                "max_drawdown_pct": results.max_drawdown_pct,
                "sharpe_ratio": results.sharpe_ratio
            },
            "period": {
                "start": results.start_date,
                "end": results.end_date,
                "days": results.duration_days
            },
            "trades": [asdict(t) for t in results.trades[:100]]  # Save first 100 trades
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")


def main():
    """Run backtest from command line."""
    parser = argparse.ArgumentParser(description="Backtest AETHER trading system")
    parser.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    parser.add_argument("--start", default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    parser.add_argument("--lot", type=float, default=0.01, help="Lot size")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min confidence threshold")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧪 AETHER BACKTESTING FRAMEWORK")
    print("="*70)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    
    # Initialize backtester
    backtester = Backtester(
        initial_balance=args.balance,
        lot_size=args.lot
    )
    
    # Generate/load data
    data = backtester.generate_synthetic_data(args.start, args.end)
    
    # Run backtest
    results = backtester.run_backtest(data, confidence_threshold=args.confidence)
    
    # Print results
    backtester.print_results(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
