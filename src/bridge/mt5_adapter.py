try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from .broker_interface import BrokerAdapter, Position, Deal
from typing import Dict, Optional
import logging
import os
import math

logger = logging.getLogger("MT5Adapter")

class MT5Adapter(BrokerAdapter):
    def __init__(self, login=None, password=None, server=None):
        self.login = int(login) if login else None
        self.password = password
        self.server = server

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 module not found. Cannot connect.")
            return False
        
        # If credentials are provided, try to initialize with them
        if self.login and self.password and self.server:
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 Login Failed: {mt5.last_error()}")
                return False
        else:
            # Fallback to default terminal state
            # Try to initialize with a specific path if standard init fails
            if not mt5.initialize():
                # Try to find the terminal path automatically
                logger.warning("Standard MT5 Init Failed. Attempting to locate terminal64.exe...")
                
                possible_paths = [
                    r"C:\Program Files\MetaTrader 5\terminal64.exe",
                    r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
                    # Add common broker paths
                    r"C:\Program Files\FTMO MetaTrader 5\terminal64.exe",
                    r"C:\Program Files\ICMarkets MetaTrader 5\terminal64.exe",
                ]
                
                success = False
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found MT5 at {path}. Attempting to initialize...")
                        if mt5.initialize(path):
                            success = True
                            break
                
                if not success:
                    error_code = mt5.last_error()
                    logger.error(f"MT5 Init Failed. Error Code: {error_code}")
                    logger.error("Troubleshooting Steps:")
                    logger.error("1. CLOSE all open MT5 terminals manually (Task Manager -> End Task).")
                    logger.error("2. Check for blocking popups (Login, News, One-Click Disclaimer).")
                    logger.error("3. Ensure 'Allow Algorithmic Trading' is ON.")
                    logger.error("4. Try running this script as Administrator.")
                    return False
        
        # Log connection details for verification
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"[MT5] Connected: Account #{account_info.login} | Server: {account_info.server}")
            logger.info(f"[MT5] Balance: ${account_info.balance:.2f} | Equity: ${account_info.equity:.2f}")
            logger.info(f"[MT5] Trade Mode: {'DEMO' if account_info.trade_mode == 1 else 'REAL' if account_info.trade_mode == 0 else 'CONTEST'}")
            logger.info(f"[MT5] Trade Allowed: {self.is_trade_allowed()}")
        else:
            logger.warning("[MT5] Could not retrieve account information")
                
        return True

    def get_market_data(self, symbol: str, timeframe: str, limit: int) -> list:
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, limit)
        if rates is None or len(rates) == 0:
            return []
        
        data = []
        for rate in rates:
            data.append({
                'time': int(rate['time']),
                'open': float(rate['open']),
                'high': float(rate['high']),
                'low': float(rate['low']),
                'close': float(rate['close']),
                'tick_volume': float(rate['tick_volume']),
                'spread': int(rate['spread']),
                'real_volume': float(rate['real_volume'])
            })
        return data

    def get_current_price(self, symbol: str) -> float:
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else 0.0

    def get_tick(self, symbol: str) -> Dict:
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol {symbol} in Market Watch")
            return None
            
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'time': tick.time,
                'flags': tick.flags
            }
        return None

    def execute_order(self, symbol, action, volume, order_type, price=None, sl=0.0, tp=0.0, magic=0, comment="", ticket=None) -> Dict:
        # ... existing logic from order_execution.py ...
        # This is where we wrap the specific MT5 code
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        
        # CRITICAL: Normalize lot size to MT5's volume_step before execution
        normalized_volume = self.normalize_lot_size(symbol, volume)
        
        # Get symbol info for deviation
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return {"ticket": None, "retcode": -1}
        
        request = {
            "symbol": symbol,
            "volume": normalized_volume,
            "type": mt5_type,
            "price": price if price else (mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid),
            "sl": sl,
            "tp": tp,
            "deviation": 20,  # Maximum price deviation in points
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if action == "CLOSE":
            # order_type already contains the correct opposing order type from caller
            # (SELL to close BUY, BUY to close SELL)
            mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
            request["action"] = mt5.TRADE_ACTION_DEAL
            request["type"] = mt5_type
            request["position"] = ticket  # Specify the position to close
            logger.info(f"[MT5] Closing position {ticket} ({order_type} order to close)")
            
        elif action == "MODIFY":
            request["action"] = mt5.TRADE_ACTION_SLTP
            request["position"] = ticket if ticket else magic
            # SL/TP are already in the request
            # Volume, type, price are ignored for SLTP modification usually, but symbol is needed
            
        else: # OPEN
            request["action"] = mt5.TRADE_ACTION_DEAL
            logger.info(f"[MT5] Sending order: {action} {order_type} {symbol} {normalized_volume} lots @ {request['price']:.5f} | SL={sl:.5f} TP={tp:.5f}")
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"[MT5] [OK] Order executed successfully - Ticket: {result.order} | Retcode: {result.retcode}")
            return {"ticket": result.order, "retcode": result.retcode}
        else:
            # Log detailed error information
            if result:
                logger.error(f"[MT5] [FAILED] Order failed: retcode={result.retcode}, comment={result.comment}")
                logger.error(f"[MT5] Request: action={action}, symbol={symbol}, volume={normalized_volume}, type={order_type}, ticket={ticket}")
            else:
                logger.error(f"[MT5] Order send returned None. Last error: {mt5.last_error()}")
            return {"ticket": None, "retcode": result.retcode if result else -1}

    def get_positions(self, symbol: Optional[str] = None) -> Optional[list]:
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        
        if positions is None:
            error_code = mt5.last_error()
            # If error is "no error" (1) but None returned, it might just be empty? 
            # No, empty returns empty tuple. None is definitely error.
            logger.warning(f"[MT5] positions_get returned None. Error: {error_code}")
            return None
            
        if len(positions) == 0:
            return []

        return [Position(
                ticket=p.ticket,
                symbol=p.symbol,
                type=p.type, # 0=BUY, 1=SELL
                volume=p.volume,
                price_open=p.price_open,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                swap=p.swap,
                comment=p.comment,
                time=p.time
            ) for p in positions]

    def get_history_deals(self, ticket: int) -> list:
        deals = mt5.history_deals_get(ticket=ticket)
        if deals:
            return [Deal(
                ticket=d.ticket,
                symbol=d.symbol,
                type=d.type,
                volume=d.volume,
                price=d.price,
                profit=d.profit,
                time=d.time
            ) for d in deals]
        return []

    def get_account_info(self) -> Dict:
        info = mt5.account_info()
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin_free": info.margin_free,
            "leverage": info.leverage
        } if info else {}

    def check_margin(self, symbol: str, volume: float, order_type: str) -> bool:
        """
        Check if there is enough margin to execute the order.
        """
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        margin = mt5.order_calc_margin(mt5_type, symbol, volume, price)
        if margin is None:
            logger.error(f"Failed to calculate margin for {symbol} {volume} lots")
            return False
            
        account_info = mt5.account_info()
        if not account_info:
            return False
            
        if account_info.margin_free < margin:
            logger.warning(f"[MARGIN] Insufficient margin! Required: ${margin:.2f}, Free: ${account_info.margin_free:.2f}")
            return False
            
        return True

    def get_max_volume(self, symbol: str, order_type: str) -> float:
        """
        Calculate maximum volume allowed by free margin.
        """
        account_info = mt5.account_info()
        if not account_info:
            return 0.0
            
        free_margin = account_info.margin_free
        mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if mt5_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        
        # Estimate margin for 1 lot
        margin_1_lot = mt5.order_calc_margin(mt5_type, symbol, 1.0, price)
        if not margin_1_lot:
            return 0.0
            
        max_vol = free_margin / margin_1_lot
        # Apply safety buffer (95% of max)
        return max_vol * 0.95

    async def close_positions(self, positions_data: list) -> dict:
        """
        ZERO-LATENCY CLOSER: Accepts full position objects/dicts to skip the lookup step.
        Executes 'Blind' close commands for maximum speed.
        """
        if not positions_data:
            return {}
            
        import asyncio
        
        # Define a wrapper to close a single ticket with retries
        def close_single_sync(pos):
            # [OPTIMIZATION] No mt5.positions_get() call here. We trust the data passed in.
            
            # Handle both object (dot notation) and dict (bracket notation)
            try:
                ticket = pos.ticket if hasattr(pos, 'ticket') else pos['ticket']
                symbol = pos.symbol if hasattr(pos, 'symbol') else pos['symbol']
                volume = pos.volume if hasattr(pos, 'volume') else pos['volume']
                p_type = pos.type if hasattr(pos, 'type') else pos['type']
                magic = pos.magic if hasattr(pos, 'magic') else pos['magic']
            except Exception as e:
                logger.error(f"Invalid position data for close: {e}")
                return {"ticket": -1, "retcode": -1, "comment": f"Invalid data: {e}"}

            # Determine close type (Opposite of open)
            order_type = mt5.ORDER_TYPE_SELL if p_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price (Fastest way)
            tick = mt5.symbol_info_tick(symbol)
            if not tick: 
                return {"ticket": ticket, "retcode": -1, "comment": "No tick data"}
                
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask

            # [OPTIMIZATION] Dynamic Deviation for Fast Closing
            # Gold moves fast, so we need wider tolerance to avoid Requotes (Latency)
            dev = 50
            if "XAU" in symbol or "GOLD" in symbol:
                dev = 500 # 50 pips tolerance for Gold (Priority: EXECUTION SPEED)
            elif "JPY" in symbol:
                dev = 100 # 10 pips for JPY pairs

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": dev,
                "magic": magic,
                "comment": "Aether FastClose",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # BLAST THE ORDER - Retry loop
            for attempt in range(3):
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[CLOSED] {ticket} at {result.price}")
                    return {"ticket": ticket, "retcode": result.retcode, "price": result.price}
                elif result.retcode in [10004, 10015, 10021]: # Requote, Invalid Price, No Money
                    # Only log warning on first failure to keep logs clean
                    if attempt == 0:
                        logger.warning(f"Close retry {ticket}: {result.comment}")
                    # Refresh price for retry
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        request['price'] = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                    continue
                else:
                    logger.error(f"CRITICAL: Failed to close {ticket}. Error: {result.comment} ({result.retcode})")
                    return {"ticket": ticket, "retcode": result.retcode, "comment": result.comment}
            
            return {"ticket": ticket, "retcode": -1, "comment": "Max retries exceeded"}

        # Get the running loop
        loop = asyncio.get_running_loop()

        # Fire ALL close requests at the exact same time using threads
        # This allows the blocking MT5 calls to run in parallel
        tasks = [
            loop.run_in_executor(None, close_single_sync, pos)
            for pos in positions_data
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Map results back to tickets (extract ticket from position data)
        tickets = [p.ticket if hasattr(p, 'ticket') else p['ticket'] for p in positions_data]
        return {ticket: result for ticket, result in zip(tickets, results)}

    def is_trade_allowed(self) -> bool:
        info = mt5.terminal_info()
        return info.trade_allowed if info else False

    def normalize_lot_size(self, symbol: str, requested_lot: float) -> float:
        """
        Normalize lot size to MT5's volume_step requirements.
        Fixes discrepancy where bot calculates 0.8 lots but MT5 accepts 0.72 lots.
        
        Args:
            symbol: Trading symbol
            requested_lot: Desired lot size from calculations
            
        Returns:
            Normalized lot size that MT5 will accept
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.warning(f"Cannot get symbol info for {symbol}, using requested lot: {requested_lot}")
            return round(requested_lot, 2)
        
        volume_min = symbol_info.get('volume_min', 0.01)
        volume_step = symbol_info.get('volume_step', 0.01)
        
        # Ensure lot is at least minimum
        if requested_lot < volume_min:
            logger.warning(f"Requested lot {requested_lot} < minimum {volume_min}, using minimum")
            return volume_min
        
        # Determine precision from volume_step
        # e.g. 0.01 -> 2 decimals, 0.1 -> 1 decimal, 1.0 -> 0 decimals
        try:
            if volume_step > 0:
                precision = int(round(-math.log10(volume_step), 0))
                if precision < 0: precision = 0
            else:
                precision = 2
        except Exception:
            precision = 2

        # Round to nearest volume_step (Standard Rounding)
        # Example: 0.1195 / 0.01 = 11.95 -> round(11.95) = 12 -> 12 * 0.01 = 0.12
        normalized = round(round(requested_lot / volume_step) * volume_step, precision)
        
        # Use epsilon for comparison
        epsilon = 0.0000001
        if abs(normalized - requested_lot) > epsilon:
            logger.info(f"[LOT NORMALIZE] {requested_lot} -> {normalized} (step: {volume_step})")
        
        return normalized

    def get_symbol_info(self, symbol: str) -> Dict:
        info = mt5.symbol_info(symbol)
        if info:
            return {
                "point": info.point,
                "digits": info.digits,
                "volume_min": info.volume_min,
                "volume_step": info.volume_step
            }
        return {}

    def disconnect(self):
        """Disconnect from the MT5 terminal."""
        if mt5:
            mt5.shutdown()
            logger.info("MT5 Disconnected")
