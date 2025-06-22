import logging
"""
GENESIS Dashboard - Active Trade Panel Component
Real-time monitoring of active trades
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from styles.dashboard_styles import tag_badge

class ActiveTradesComponent:
    def detect_confluence_patterns(self, market_data: dict) -> float:
            """GENESIS Pattern Intelligence - Detect confluence patterns"""
            confluence_score = 0.0

            # Simple confluence calculation
            if market_data.get('trend_aligned', False):
                confluence_score += 0.3
            if market_data.get('support_resistance_level', False):
                confluence_score += 0.3
            if market_data.get('volume_confirmation', False):
                confluence_score += 0.2
            if market_data.get('momentum_aligned', False):
                confluence_score += 0.2

            emit_telemetry("active_trades", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("active_trades", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "active_trades",
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
                print(f"Emergency stop error in active_trades: {e}")
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_bus = self._get_event_bus()
        
    def _get_event_bus(self):
        # Auto-injected EventBus connection
        try:
            from event_bus_manager import EventBusManager


# <!-- @GENESIS_MODULE_END: active_trades -->


# <!-- @GENESIS_MODULE_START: active_trades -->
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """
    Component for displaying real-time active trades
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.active_trade_config = config["active_trade_panel"]
        self.refresh_rate = config["refresh_rate"]["trades"]
        self.last_updated = datetime.now()
        self.trade_data = []
    
    def load_active_trades(self):
        """Load active trades from ExecutionEngine and TradeJournalEngine"""
        try:
            active_trades = []
            
            # Look for trade journal entries in logs
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Try today's and yesterday's logs
            journal_files = [
                "logs/journal/trades_active.jsonl",
                f"logs/journal/trades_{today}.jsonl",
                f"logs/journal/trades_{yesterday}.jsonl"
            ]
            
            for file_path in journal_files:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                trade = json.loads(line.strip())
                                # Check if it's an active trade
                                if trade.get("status") == "active":
                                    # Calculate current P/L if we have enough data
                                    if "entry_price" in trade and "current_price" in trade:
                                        entry = float(trade["entry_price"])
                                        current = float(trade["current_price"])
                                        direction = trade.get("direction", "").lower()
                                        
                                        if direction == "buy":
                                            pnl_pct = (current - entry) / entry * 100
                                            pnl_pips = (current - entry) * 10000  # Assuming 4 decimal places
                                        elif direction == "sell":
                                            pnl_pct = (entry - current) / entry * 100
                                            pnl_pips = (entry - current) * 10000  # Assuming 4 decimal places
                                        else:
                                            pnl_pct = 0
                                            pnl_pips = 0
                                        
                                        trade["pnl_pct"] = round(pnl_pct, 2)
                                        trade["pnl_pips"] = round(pnl_pips, 1)
                                    
                                    # Calculate trade duration
                                    if "entry_time" in trade:
                                        try:
                                            entry_time = datetime.fromisoformat(trade["entry_time"])
                                            now = datetime.now()
                                            duration = now - entry_time
                                            hours, remainder = divmod(duration.total_seconds(), 3600)
                                            minutes, seconds = divmod(remainder, 60)
                                            trade["duration"] = f"{int(hours)}h {int(minutes)}m"
                                        except:
                                            trade["duration"] = "Unknown"
                                    
                                    # Calculate R:R ratio
                                    if "entry_price" in trade and "sl" in trade and "tp" in trade:
                                        entry = float(trade["entry_price"])
                                        sl = float(trade["sl"])
                                        tp = float(trade["tp"])
                                        direction = trade.get("direction", "").lower()
                                        
                                        if direction == "buy":
                                            risk = entry - sl
                                            reward = tp - entry
                                        elif direction == "sell":
                                            risk = sl - entry
                                            reward = entry - tp
                                        else:
                                            risk = 0
                                            reward = 0
                                        
                                        if risk > 0:
                                            trade["rr_ratio"] = round(reward / risk, 2)
                                        else:
                                            trade["rr_ratio"] = 0
                                    
                                    active_trades.append(trade)
                            except Exception as e:
                                continue
                                
            self.trade_data = active_trades
            self.last_updated = datetime.now()
            return active_trades
            
        except Exception as e:
            st.error(f"Error loading active trades: {str(e)}")
            return []
    
    def render(self):
        """Render the active trade panel"""
        st.markdown('<div class="main-title">Active Trades</div>', unsafe_allow_html=True)
        
        # Get current active trades
        active_trades = self.load_active_trades()
        
        # Display last updated time
        st.markdown(f'<div class="last-update">Last updated: {self.last_updated.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
        
        # Show filters
        with st.expander("Filter & Sort Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                direction_filter = st.multiselect(
                    "Direction", 
                    options=["Buy", "Sell"],
                    default=[]
                )
            
            with col2:
                symbol_filter = st.multiselect(
                    "Symbol",
                    options=list(set([t.get("symbol", "Unknown") for t in active_trades])),
                    default=[]
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    options=["Entry Time", "Symbol", "P&L", "Duration", "R:R"],
                    index=0
                )
                
            tag_filters = st.multiselect(
                "Tags",
                options=self.active_trade_config.get("tags", []),
                default=[]
            )
        
        # Apply filters
        filtered_trades = active_trades
        
        if direction_filter:
            filtered_trades = [t for t in filtered_trades if t.get("direction", "").lower() in [d.lower() for d in direction_filter]]
        
        if symbol_filter:
            filtered_trades = [t for t in filtered_trades if t.get("symbol", "") in symbol_filter]
        
        if tag_filters:
            filtered_trades = [t for t in filtered_trades if any(tag.lower() in (t.get("tags", []) or []) for tag in tag_filters)]
        
        # Apply sorting
        if sort_by == "Entry Time":
            filtered_trades = sorted(filtered_trades, key=lambda t: t.get("entry_time", ""), reverse=True)
        elif sort_by == "Symbol":
            filtered_trades = sorted(filtered_trades, key=lambda t: t.get("symbol", ""))
        elif sort_by == "P&L":
            filtered_trades = sorted(filtered_trades, key=lambda t: t.get("pnl_pct", 0), reverse=True)
        elif sort_by == "Duration":
            filtered_trades = sorted(filtered_trades, key=lambda t: t.get("entry_time", ""))
        elif sort_by == "R:R":
            filtered_trades = sorted(filtered_trades, key=lambda t: t.get("rr_ratio", 0), reverse=True)
        
        # Check if we have active trades
        if not filtered_trades:
            st.info("No active trades at the moment.")
            return
        
        # Convert to DataFrame for easier display
        df = pd.DataFrame(filtered_trades)
        
        # Reorder and select columns based on config
        columns_to_display = [col for col in self.active_trade_config["columns"] if col in df.columns]
        
        if columns_to_display:
            df = df[columns_to_display]
        
        # Display as table with conditional formatting
        st.dataframe(
            df.style.apply(
                lambda x: ['background-color: rgba(34,169,34,0.1)' if x.name == 'pnl_pct' and v > 0 else 
                          'background-color: rgba(255,65,54,0.1)' if x.name == 'pnl_pct' and v < 0 else 
                          '' for v in x],
                axis=0
            ),
            height=400
        )
        
        # Display trade statistics
        if filtered_trades:
            st.markdown('<div class="subtitle">Trade Statistics</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Active Trades", len(filtered_trades))
            
            with col2:
                buy_count = len([t for t in filtered_trades if t.get("direction", "").lower() == "buy"])
                sell_count = len([t for t in filtered_trades if t.get("direction", "").lower() == "sell"])
                st.metric("Buy/Sell Ratio", f"{buy_count}/{sell_count}")
            
            with col3:
                avg_rr = sum([t.get("rr_ratio", 0) for t in filtered_trades]) / len(filtered_trades) if filtered_trades else 0
                st.metric("Average R:R", f"{avg_rr:.2f}")
            
            with col4:
                profitable_trades = len([t for t in filtered_trades if t.get("pnl_pct", 0) > 0])
                win_rate = profitable_trades / len(filtered_trades) if filtered_trades else 0
                st.metric("Current Win Rate", f"{win_rate*100:.1f}%")


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
