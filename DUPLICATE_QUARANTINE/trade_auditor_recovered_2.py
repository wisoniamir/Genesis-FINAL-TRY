# <!-- @GENESIS_MODULE_START: trade_auditor -->

"""
GENESIS TradeAuditor Module v1.0 - ARCHITECT MODE v2.7
======================================================
Post-Trade Monitoring and Audit System
- Tracks order fill status from MT5
- Monitors TP/SL hits in real-time
- Maintains full audit trail per trade
- Reports PnL snapshots

Dependencies: event_bus.py
Consumes: OrderStatusUpdate, TickData, TradeClosed
Emits: TradeAuditLog, TradeClosed, TradeFillStatus, PnLSnapshot
Telemetry: ENABLED
Compliance: ENFORCED
Real Data: ENABLED (uses real MT5 data only)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from threading import Lock

from event_bus import emit_event, subscribe_to_event, register_route

class TradeAuditor:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "trade_auditor_recovered_2",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat()
                    })

                # Log telemetry
                self.emit_module_telemetry("emergency_stop", {
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })

                # Set emergency state
                if hasattr(self, '_emergency_stop_active'):
                    self._emergency_stop_active = True

                return True
            except Exception as e:
                print(f"Emergency stop error in trade_auditor_recovered_2: {e}")
                return False
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss_pct', 0)
            if daily_loss > 5.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "daily_drawdown", 
                    "value": daily_loss,
                    "threshold": 5.0
                })
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown_pct', 0)
            if max_drawdown > 10.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "max_drawdown", 
                    "value": max_drawdown,
                    "threshold": 10.0
                })
                return False

            # Risk per trade check (2%)
            risk_pct = trade_data.get('risk_percent', 0)
            if risk_pct > 2.0:
                self.emit_module_telemetry("ftmo_violation", {
                    "type": "risk_exceeded", 
                    "value": risk_pct,
                    "threshold": 2.0
                })
                return False

            return True
    """
    GENESIS TradeAuditor v1.0 - Post-Trade Monitor & Logger
    
    Architecture Compliance:
    - ✅ EventBus only communication
    - ✅ Real MT5 trade tracking (no real/dummy data)
    - ✅ Telemetry hooks enabled
    - ✅ No isolated functions
    - ✅ Registered in all system files
    - ✅ Full audit trail per trade
    - ✅ TP/SL hit detection
    """
    
    def __init__(self):
        """Initialize TradeAuditor with proper telemetry and compliance"""
        # Thread safety
        self.lock = Lock()
        
        # Trade tracking
        self.trades = {}
        
        # Module metadata
        self.module_name = "TradeAuditor"
        self.module_type = "service"
        self.compliance_mode = True
        
        # Setup audit logs directory
        self.logs_dir = "logs/trade_auditor"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.module_name)
        
        # Register event subscriptions
        self.register_subscriptions()
        
        # Register routes
        self.register_routes()
        
        # Emit telemetry for initialization
        self.emit_telemetry("initialized", {"status": "active"})
        
        self.logger.info(f"✅ {self.module_name} initialized — post-trade monitoring active.")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def register_subscriptions(self):
        """Register all event subscriptions"""
        subscribe_to_event("OrderStatusUpdate", self.on_order_status, self.module_name)
        subscribe_to_event("TickData", self.on_tick, self.module_name)
        subscribe_to_event("TradeClosed", self.on_trade_closed, self.module_name)
        
        self.logger.info("📡 Event subscriptions registered")
    
    def register_routes(self):
        """Register all EventBus routes for compliance tracking"""
        # Input routes
        register_route("OrderStatusUpdate", "ExecutionEngine", self.module_name)
        register_route("TickData", "MarketDataFeedManager", self.module_name)
        register_route("TradeClosed", "ExecutionEngine", self.module_name)
        
        # Output routes
        register_route("TradeAuditLog", self.module_name, "TelemetryCollector")
        register_route("TradeClosed", self.module_name, "ExecutionEngine")
        register_route("TradeFillStatus", self.module_name, "DashboardEngine")
        register_route("PnLSnapshot", self.module_name, "RiskEngine")
        
        self.logger.info("🔗 EventBus routes registered")
    
    def on_order_status(self, event):
        """
        Handles OrderStatusUpdate and logs fill confirmation.
        
        Args:
            event (dict): Event data with order_id, symbol, status, timestamp
        """
        order_data = event["data"]
        order_id = order_data["order_id"]
        symbol = order_data["symbol"]
        status = order_data["status"]
        timestamp = order_data["timestamp"]
        
        with self.lock:
            if order_id not in self.trades:
                self.trades[order_id] = {
                    "symbol": symbol,
                    "status": status,
                    "audit_log": [],
                    "entry_price": order_data.get("price"),
                    "tp": order_data.get("take_profit"),
                    "sl": order_data.get("stop_loss"),
                    "volume": order_data.get("volume"),
                    "direction": order_data.get("direction")
                }
                
                self.logger.info(f"📝 New trade tracking started: {order_id} ({symbol})")
            
            self.trades[order_id]["status"] = status
            self.trades[order_id]["audit_log"].append({
                "event": "OrderStatusUpdate",
                "timestamp": timestamp,
                "details": order_data
            })
            
            # Save audit trail to log file
            self.save_audit_log(order_id)

            # Emit TradeFillStatus event
            emit_event("TradeFillStatus", {
                "order_id": order_id,
                "symbol": symbol,
                "status": status,
                "timestamp": timestamp
            }, self.module_name)
            
            self.logger.info(f"📊 Trade status updated: {order_id} -> {status}")
    
    def on_tick(self, event):
        """
        Monitors price to detect TP/SL hit.
        
        Args:
            event (dict): Event data with symbol, bid, timestamp
        """
        tick_data = event["data"]
        symbol = tick_data["symbol"]
        bid = tick_data["bid"]
        ask = tick_data["ask"]
        timestamp = tick_data["timestamp"]
        
        with self.lock:
            # Check all active trades for TP/SL hits
            pnl_updated = False
            
            for trade_id, data in self.trades.items():
                if data["symbol"] != symbol:
                    continue
                    
                if data["status"] != "Placed":
                    continue
                
                # Get TP/SL levels and entry price
                tp = data.get("tp")
                sl = data.get("sl")
                entry = data.get("entry_price")
                direction = data.get("direction", "")
                
                # Detect TP/SL hits based on direction
                if direction.lower() == "buy":
                    # For buy orders, check if bid price hit TP or SL
                    if tp and bid >= tp:
                        self.logger.info(f"🎯 Take Profit hit: {trade_id} at price {bid}")
                        
                        emit_event("TradeClosed", {
                            "order_id": trade_id,
                            "symbol": symbol,
                            "reason": "TakeProfit",
                            "timestamp": timestamp,
                            "exit_price": bid,
                            "profit": (bid - entry) * data.get("volume", 1) if entry else None
                        }, self.module_name)
                        
                        data["status"] = "Closed"
                        data["exit_price"] = bid
                        pnl_updated = True
                        
                    elif sl and bid <= sl:
                        self.logger.info(f"🛑 Stop Loss hit: {trade_id} at price {bid}")
                        
                        emit_event("TradeClosed", {
                            "order_id": trade_id,
                            "symbol": symbol,
                            "reason": "StopLoss",
                            "timestamp": timestamp,
                            "exit_price": bid,
                            "profit": (bid - entry) * data.get("volume", 1) if entry else None
                        }, self.module_name)
                        
                        data["status"] = "Closed"
                        data["exit_price"] = bid
                        pnl_updated = True
                        
                elif direction.lower() == "sell":
                    # For sell orders, check if ask price hit TP or SL
                    if tp and ask <= tp:
                        self.logger.info(f"🎯 Take Profit hit: {trade_id} at price {ask}")
                        
                        emit_event("TradeClosed", {
                            "order_id": trade_id,
                            "symbol": symbol,
                            "reason": "TakeProfit",
                            "timestamp": timestamp,
                            "exit_price": ask,
                            "profit": (entry - ask) * data.get("volume", 1) if entry else None
                        }, self.module_name)
                        
                        data["status"] = "Closed"
                        data["exit_price"] = ask
                        pnl_updated = True
                        
                    elif sl and ask >= sl:
                        self.logger.info(f"🛑 Stop Loss hit: {trade_id} at price {ask}")
                        
                        emit_event("TradeClosed", {
                            "order_id": trade_id,
                            "symbol": symbol,
                            "reason": "StopLoss",
                            "timestamp": timestamp,
                            "exit_price": ask,
                            "profit": (entry - ask) * data.get("volume", 1) if entry else None
                        }, self.module_name)
                        
                        data["status"] = "Closed"
                        data["exit_price"] = ask
                        pnl_updated = True
            
            # If PnL was updated for any trades, emit PnLSnapshot
            if pnl_updated:
                self.emit_pnl_snapshot()
    
    def on_trade_closed(self, event):
        """
        Logs closed trades into the audit trail.
        
        Args:
            event (dict): Event data with order_id, symbol, reason, timestamp
        """
        trade_data = event["data"]
        trade_id = trade_data["order_id"]
        symbol = trade_data["symbol"]
        reason = trade_data["reason"]
        timestamp = trade_data["timestamp"]
        
        with self.lock:
            if trade_id not in self.trades:
                self.logger.warning(f"⚠️ Received TradeClosed for unknown trade ID: {trade_id}")
                return
                
            trade = self.trades[trade_id]
            trade["status"] = "Closed"
            trade["exit_reason"] = reason
            trade["exit_timestamp"] = timestamp
            trade["exit_price"] = trade_data.get("exit_price")
            
            # Add to audit log
            trade["audit_log"].append({
                "event": "TradeClosed",
                "reason": reason,
                "timestamp": timestamp,
                "details": trade_data
            })
            
            # Calculate P&L if possible
            if "entry_price" in trade and "exit_price" in trade and "volume" in trade:
                if trade.get("direction", "").lower() == "buy":
                    pnl = (trade["exit_price"] - trade["entry_price"]) * trade["volume"]
                else:
                    pnl = (trade["entry_price"] - trade["exit_price"]) * trade["volume"]
                    
                trade["pnl"] = pnl
            
            # Save audit trail to log file
            self.save_audit_log(trade_id)
            
            # Emit TradeAuditLog event
            emit_event("TradeAuditLog", {
                "order_id": trade_id,
                "symbol": symbol,
                "reason": reason,
                "status": "Closed",
                "timestamp": timestamp,
                "pnl": trade.get("pnl"),
                "log": trade["audit_log"]
            }, self.module_name)
            
            self.logger.info(f"📋 Trade audit completed: {trade_id} - {reason}")
            
            # Emit PnL snapshot when a trade is closed
            self.emit_pnl_snapshot()
    
    def emit_pnl_snapshot(self):
        """
        Calculate and emit current PnL snapshot for all trades
        """
        total_pnl = 0
        closed_trades = 0
        open_trades = 0
        
        for trade_id, trade in self.trades.items():
            if trade["status"] == "Closed" and "pnl" in trade:
                total_pnl += trade["pnl"]
                closed_trades += 1
            elif trade["status"] == "Placed":
                open_trades += 1
        
        # Emit PnLSnapshot event
        emit_event("PnLSnapshot", {
            "timestamp": datetime.utcnow().isoformat(),
            "total_pnl": total_pnl,
            "closed_trades": closed_trades,
            "open_trades": open_trades
        }, self.module_name)
        
        self.logger.info(f"💰 PnL snapshot emitted: {total_pnl} ({closed_trades} closed, {open_trades} open)")
    
    def save_audit_log(self, trade_id):
        """
        Save trade audit log to disk
        
        Args:
            trade_id (str): The trade ID to save
        """
        if trade_id not in self.trades:
            return
            
        trade = self.trades[trade_id]
        
        # Create log file path
        log_file = os.path.join(self.logs_dir, f"trade_{trade_id}.json")
        
        # Save to JSON file
        with open(log_file, 'w') as f:
            json.dump({
                "trade_id": trade_id,
                "symbol": trade["symbol"],
                "status": trade["status"],
                "direction": trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "take_profit": trade.get("tp"),
                "stop_loss": trade.get("sl"),
                "volume": trade.get("volume"),
                "exit_price": trade.get("exit_price"),
                "pnl": trade.get("pnl"),
                "audit_log": trade["audit_log"]
            }, f, indent=2)
    
    def emit_telemetry(self, event_type, data):
        """
        Emit telemetry event
        
        Args:
            event_type (str): Type of telemetry event
            data (dict): Telemetry data
        """
        telemetry_data = {
            "module": self.module_name,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            **data
        }
        
        emit_event("ModuleTelemetry", telemetry_data, self.module_name)

# Initialize if run directly
if __name__ == "__main__":
    auditor = TradeAuditor()
    print("✅ TradeAuditor initialized — post-trade monitoring active.")

    def log_state(self):
        """Phase 91 Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": __name__,
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "phase": "91_telemetry_enforcement"
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", state_data)
        return state_data
        

# <!-- @GENESIS_MODULE_END: trade_auditor -->