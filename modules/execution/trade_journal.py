import logging

# 🔗 GENESIS EventBus Integration - Auto-injected by Orphan Recovery Engine
from datetime import datetime
import json

class TradeJournalEventBusIntegration:
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

            emit_telemetry("trade_journal", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("trade_journal", "position_calculated", {
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
                        "module": "trade_journal",
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
                print(f"Emergency stop error in trade_journal: {e}")
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
    """EventBus integration for trade_journal"""
    
    def __init__(self):
        self.module_id = "trade_journal"
        self.event_routes = []
        
    def emit_event(self, event_type, data):
        """Emit event to EventBus"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_id,
            "event_type": event_type,
            "data": data
        }
        print(f"🔗 EVENTBUS EMIT: {event}")
        
    def emit_telemetry(self, metric_name, value):
        """Emit telemetry data"""
        telemetry = {
            "timestamp": datetime.now().isoformat(),
            "module": self.module_id,
            "metric": metric_name,
            "value": value
        }
        print(f"📊 TELEMETRY: {telemetry}")

# Auto-instantiate EventBus integration
trade_journal_eventbus = TradeJournalEventBusIntegration()

"""
GENESIS Dashboard - Trade Journal Timeline Component
Visualizing trade history with interactive timeline
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TradeJournalComponent:
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

            emit_telemetry("trade_journal", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("trade_journal", "position_calculated", {
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
                        "module": "trade_journal",
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
                print(f"Emergency stop error in trade_journal: {e}")
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


# <!-- @GENESIS_MODULE_END: trade_journal -->


# <!-- @GENESIS_MODULE_START: trade_journal -->
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """
    Component for displaying trade journal with timeline and metrics
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.journal_config = config["trade_journal"]
        self.refresh_rate = config["refresh_rate"]["trades"]
        self.last_updated = datetime.now()
        self.trades = []
    
    def load_trade_history(self, timespan="7d"):
        """Load trade history from TradeJournalEngine"""
        try:
            trades = []
            
            # Determine the date range based on timespan
            if timespan == "1d":
                start_date = datetime.now() - timedelta(days=1)
            elif timespan == "7d":
                start_date = datetime.now() - timedelta(days=7)
            elif timespan == "30d":
                start_date = datetime.now() - timedelta(days=30)
            elif timespan == "90d":
                start_date = datetime.now() - timedelta(days=90)
            else:  # all
                start_date = datetime.now() - timedelta(days=365)  # Just a very long time
            
            # Look for journal files
            log_path = "logs/journal/"
            if os.path.exists(log_path):
                for file_name in os.listdir(log_path):
                    if file_name.endswith(".jsonl") and file_name.startswith("trades_"):
                        try:
                            with open(os.path.join(log_path, file_name), "r") as f:
                                for line in f:
                                    trade = json.loads(line.strip())
                                    
                                    # Check timestamp against timespan
                                    if "close_time" in trade and trade["close_time"]:
                                        try:
                                            close_time = datetime.fromisoformat(trade["close_time"])
                                            if close_time >= start_date:
                                                trades.append(trade)
                                        except:
                                            # If we can't parse the timestamp, include anyway
                                            trades.append(trade)
                                    elif "entry_time" in trade:
                                        try:
                                            entry_time = datetime.fromisoformat(trade["entry_time"])
                                            if entry_time >= start_date:
                                                trades.append(trade)
                                        except:
                                            # If we can't parse the timestamp, include anyway
                                            trades.append(trade)
                                    else:
                                        trades.append(trade)
                        except:
                            continue
            
            # Sort by entry time
            trades = sorted(trades, key=lambda t: t.get("entry_time", ""))
            
            self.trades = trades
            self.last_updated = datetime.now()
            return trades
            
        except Exception as e:
            st.error(f"Error loading trade history: {str(e)}")
            return []
    
    def create_timeline_chart(self, trades):
        """Create timeline chart of trades"""
        # Extract data
        data = []
        cumulative_pnl = 0
        
        for trade in trades:
            # Skip trades without required data
            if "entry_time" not in trade or "symbol" not in trade or "outcome" not in trade:
                continue
            
            if "profit_loss" in trade:
                try:
                    pnl = float(trade["profit_loss"])
                    cumulative_pnl += pnl
                except:
                    pnl = 0
            else:
                pnl = 0
            
            data.append({
                "entry_time": trade.get("entry_time"),
                "close_time": trade.get("close_time", trade.get("entry_time")),
                "symbol": trade.get("symbol"),
                "direction": trade.get("direction", "unknown"),
                "outcome": trade.get("outcome", "unknown"),
                "profit_loss": pnl,
                "cumulative_pnl": cumulative_pnl,
                "rr_achieved": trade.get("rr_achieved", 0)
            })
        
        if not data:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Convert timestamps to datetime
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["close_time"] = pd.to_datetime(df["close_time"])
        
        # Create subplots with 2 rows
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Trade Outcomes", "Cumulative P&L")
        )
        
        # Trade markers
        for outcome in ["win", "loss", "breakeven", "unknown"]:
            mask = df["outcome"] == outcome
            
            if not any(mask):
                continue
                
            color = "#22A922" if outcome == "win" else "#FF4136" if outcome == "loss" else "#8A8A8A"
            
            fig.add_trace(
                go.Scatter(
                    x=df[mask]["close_time"],
                    y=[1] * sum(mask),  # All points at y=1
                    mode="markers",
                    marker=dict(size=12, color=color, symbol="circle"),
                    name=outcome.capitalize(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>" +
                        "Symbol: %{customdata[1]}<br>" +
                        "Direction: %{customdata[2]}<br>" +
                        "P&L: %{customdata[3]:.2f}<br>" +
                        "R:R: %{customdata[4]:.2f}<br>" +
                        "%{customdata[5]}"
                    ),
                    customdata=list(zip(
                        df[mask]["close_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                        df[mask]["symbol"],
                        df[mask]["direction"],
                        df[mask]["profit_loss"],
                        df[mask]["rr_achieved"],
                        df[mask]["outcome"].apply(lambda x: f"Outcome: {x.capitalize()}")
                    ))
                ),
                row=1, col=1
            )
        
        # Cumulative P&L line
        fig.add_trace(
            go.Scatter(
                x=df["close_time"],
                y=df["cumulative_pnl"],
                mode="lines+markers",
                line=dict(width=2, color="#00BFFF"),
                name="Cumulative P&L",
                hovertemplate="<b>%{x}</b><br>Cumulative P&L: %{y:.2f}"
            ),
            row=2, col=1
        )
        
        # Add horizontal line at y=0 for P&L
        fig.add_shape(
            type="line",
            x0=df["close_time"].min(),
            x1=df["close_time"].max(),
            y0=0,
            y1=0,
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            template="plotly_dark",
            plot_bgcolor="#1A1E24",
            paper_bgcolor="#1A1E24",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        
        # Update y-axes
        fig.update_yaxes(
            showticklabels=False, 
            row=1, col=1,
            fixedrange=True
        )
        
        fig.update_yaxes(
            title_text="P&L", 
            row=2, col=1,
            gridcolor="rgba(255, 255, 255, 0.1)"
        )
        
        # Update x-axis
        fig.update_xaxes(
            title_text="Date/Time", 
            row=2, col=1,
            gridcolor="rgba(255, 255, 255, 0.1)"
        )
        
        return fig
    
    def create_summary_metrics(self, trades):
        """Create summary metrics for the trades"""
        metrics = {
            "total_trades": len(trades),
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "pnl_sum": 0,
            "max_win": 0,
            "max_loss": 0,
            "avg_trade": 0,
            "win_rate": 0
        }
        
        for trade in trades:
            outcome = trade.get("outcome", "unknown")
            
            if outcome == "win":
                metrics["wins"] += 1
                pnl = float(trade.get("profit_loss", 0))
                metrics["pnl_sum"] += pnl
                metrics["max_win"] = max(metrics["max_win"], pnl)
            elif outcome == "loss":
                metrics["losses"] += 1
                pnl = float(trade.get("profit_loss", 0))
                metrics["pnl_sum"] += pnl
                metrics["max_loss"] = min(metrics["max_loss"], pnl)
            elif outcome == "breakeven":
                metrics["breakeven"] += 1
                pnl = float(trade.get("profit_loss", 0))
                metrics["pnl_sum"] += pnl
        
        if metrics["total_trades"] > 0:
            metrics["avg_trade"] = metrics["pnl_sum"] / metrics["total_trades"]
            
            if (metrics["wins"] + metrics["losses"]) > 0:
                metrics["win_rate"] = metrics["wins"] / (metrics["wins"] + metrics["losses"]) * 100
        
        return metrics
    
    def render(self):
        """Render the trade journal timeline"""
        st.markdown('<div class="main-title">Trade Journal Timeline</div>', unsafe_allow_html=True)
        
        # Add timespan selector
        col1, col2 = st.columns([3, 1])
        with col2:
            timespan = st.selectbox(
                "Timeframe",
                options=self.journal_config["timeline_view"]["available_timespans"],
                index=self.journal_config["timeline_view"]["available_timespans"].index(
                    self.journal_config["timeline_view"]["default_timespan"]
                )
            )
        
        # Get trade history
        trades = self.load_trade_history(timespan)
        
        # Display last updated time
        st.markdown(f'<div class="last-update">Last updated: {self.last_updated.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
        
        if not trades:
            st.info("No trades found for the selected timeframe.")
            return
        
        # Create timeline chart
        fig = self.create_timeline_chart(trades)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to create timeline chart.")
        
        # Display trade metrics
        metrics = self.create_summary_metrics(trades)
        
        st.markdown('<div class="subtitle">Trade Statistics</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", metrics["total_trades"])
        
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        
        with col3:
            st.metric("Net P&L", f"{metrics['pnl_sum']:.2f}")
        
        with col4:
            st.metric("Average Trade", f"{metrics['avg_trade']:.2f}")
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Wins", metrics["wins"])
        
        with col2:
            st.metric("Losses", metrics["losses"])
        
        with col3:
            st.metric("Max Win", f"{metrics['max_win']:.2f}")
        
        with col4:
            st.metric("Max Loss", f"{metrics['max_loss']:.2f}")
        
        # Show raw trade data in expandable table
        with st.expander("View Raw Trade Data", expanded=False):
            if trades:
                df = pd.DataFrame(trades)
                st.dataframe(df, height=300)


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
