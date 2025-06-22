import logging
# <!-- @GENESIS_MODULE_START: backtest_explorer -->

from event_bus import EventBus

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
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

                emit_telemetry("backtest_explorer_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("backtest_explorer_recovered_1", "position_calculated", {
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
                            "module": "backtest_explorer_recovered_1",
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
                    print(f"Emergency stop error in backtest_explorer_recovered_1: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "backtest_explorer_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("backtest_explorer_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in backtest_explorer_recovered_1: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


"""
GENESIS Dashboard - Backtest Explorer Component
Interactive backtest results visualization and analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestExplorerComponent:
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

            emit_telemetry("backtest_explorer_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("backtest_explorer_recovered_1", "position_calculated", {
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
                        "module": "backtest_explorer_recovered_1",
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
                print(f"Emergency stop error in backtest_explorer_recovered_1: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "backtest_explorer_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("backtest_explorer_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in backtest_explorer_recovered_1: {e}")
    """
    Component for displaying backtest results with interactive visualizations
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self._emit_startup_telemetry()
        self.config = config
        self.backtest_config = config["backtest_explorer"]
        self.refresh_rate = config["refresh_rate"]["signals"]
        self.last_updated = datetime.now()
    
    def load_backtest_results(self):
        """Load backtest results from BacktestEngine"""
        try:
            results = []
            
            # Look for backtest files
            log_path = "logs/backtest_results/"
            if os.path.exists(log_path):
                for file_name in os.listdir(log_path):
                    if file_name.endswith(".jsonl"):
                        try:
                            with open(os.path.join(log_path, file_name), "r") as f:
                                for line in f:
                                    backtest = json.loads(line.strip())
                                    results.append(backtest)
                        except:
                            continue
            
            # Sort by timestamp
            results = sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            self.last_updated = datetime.now()
            return results
            
        except Exception as e:
            st.error(f"Error loading backtest results: {str(e)}")
            return []
    
    def create_equity_curve(self, backself.event_bus.request('data:live_feed')):
        """Create equity curve visualization"""
        # Check if we have trades data
        if not backself.event_bus.request('data:live_feed') or "trades" not in backself.event_bus.request('data:live_feed'):
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
        
        trades = backself.event_bus.request('data:live_feed')["trades"]
        
        # Create dataframe for equity curve
        trade_data = []
        equity = 100000  # Starting equity
        
        for trade in trades:
            trade_time = trade.get("close_time", "")
            pnl = float(trade.get("profit_loss", 0))
            
            equity += pnl
            
            trade_data.append({
                "time": trade_time,
                "equity": equity,
                "pnl": pnl,
                "symbol": trade.get("symbol", "Unknown"),
                "direction": trade.get("direction", "Unknown"),
                "outcome": trade.get("outcome", "Unknown")
            })
        
        # Create dataframe
        df = pd.DataFrame(trade_data)
        
        # Convert time to datetime
        df["time"] = pd.to_datetime(df["time"])
        
        # Create subplots: equity curve and drawdown
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Equity Curve", "Drawdown"),
            row_heights=[0.7, 0.3]
        )
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["equity"],
                mode="lines",
                name="Equity",
                line=dict(width=2, color="#00BFFF"),
                hovertemplate="<b>%{x}</b><br>Equity: %{y:.2f}"
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        if not df.empty:
            df["peak"] = df["equity"].cummax()
            df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"] * 100
            
            # Add drawdown chart
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df["drawdown"],
                    mode="lines",
                    name="Drawdown %",
                    line=dict(width=2, color="#FF4136"),
                    fill="tozeroy",
                    fillcolor="rgba(255, 65, 54, 0.2)",
                    hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.2f}%"
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            template="plotly_dark",
            plot_bgcolor="#1A1E24",
            paper_bgcolor="#1A1E24",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified"
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Equity", 
            row=1, col=1,
            gridcolor="rgba(255, 255, 255, 0.1)"
        )
        
        fig.update_yaxes(
            title_text="Drawdown %", 
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
    
    def create_trade_cluster_viz(self, backself.event_bus.request('data:live_feed')):
        """Create trade cluster visualization"""
        if not backself.event_bus.request('data:live_feed') or "trades" not in backself.event_bus.request('data:live_feed'):
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
        
        trades = backself.event_bus.request('data:live_feed')["trades"]
        
        # Extract data for visualization
        trade_data = []
        
        for trade in trades:
            pnl = float(trade.get("profit_loss", 0))
            
            trade_data.append({
                "symbol": trade.get("symbol", "Unknown"),
                "direction": trade.get("direction", "Unknown"),
                "outcome": trade.get("outcome", "Unknown"),
                "profit_loss": pnl,
                "rr_achieved": float(trade.get("rr_achieved", 0)),
                "strategy": trade.get("strategy_type", "Unknown"),
                "timeframe": trade.get("timeframe", "Unknown"),
                "time_of_day": pd.to_datetime(trade.get("entry_time", "")).hour if trade.get("entry_time") else 0
            })
        
        df = pd.DataFrame(trade_data)
        
        # Create scatter plot of trades
        fig = px.scatter(
            df,
            x="time_of_day",
            y="profit_loss",
            color="outcome",
            symbol="direction",
            size="rr_achieved",
            size_max=15,
            hover_data=["symbol", "strategy", "timeframe"],
            labels={"time_of_day": "Hour of Day (0-23)", "profit_loss": "Profit/Loss"},
            color_discrete_map={"win": "#22A922", "loss": "#FF4136", "breakeven": "#8A8A8A", "Unknown": "#8A8A8A"},
            title="Trade Clusters by Time of Day"
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            template="plotly_dark",
            plot_bgcolor="#1A1E24",
            paper_bgcolor="#1A1E24",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="closest"
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            range=[-0.5, 23.5],
            tickvals=list(range(0, 24, 3))
        )
        
        fig.update_yaxes(
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.5)",
            zerolinewidth=1
        )
        
        return fig
    
    def create_performance_metrics(self, backself.event_bus.request('data:live_feed')):
        """Create performance metrics table and visualizations"""
        if not backself.event_bus.request('data:live_feed') or "performance_metrics" not in backself.event_bus.request('data:live_feed'):
            self.event_bus.emit('error:fallback_triggered', {'module': __name__})
        return self.event_bus.request('data:default_value')
            
        metrics = backself.event_bus.request('data:live_feed')["performance_metrics"]
        
        # Format metrics for display
        formatted_metrics = {
            "Total Trades": metrics.get("total_trades", 0),
            "Win Rate": f"{metrics.get('win_rate', 0):.2f}%",
            "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Max Drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            "Recovery Factor": f"{metrics.get('recovery_factor', 0):.2f}",
            "Avg. Trade": f"{metrics.get('average_trade', 0):.2f}",
            "Avg. Win": f"{metrics.get('average_win', 0):.2f}",
            "Avg. Loss": f"{metrics.get('average_loss', 0):.2f}",
            "Win/Loss Ratio": f"{metrics.get('win_loss_ratio', 0):.2f}",
            "Expectancy": f"{metrics.get('expectancy', 0):.2f}",
            "Avg. Hold Time": f"{metrics.get('avg_hold_time', 0)} min"
        }
        
        return formatted_metrics
    
    def render(self):
        """Render the backtest explorer"""
        st.markdown('<div class="main-title">Backtest Explorer</div>', unsafe_allow_html=True)
        
        # Get backtest results
        backtest_results = self.load_backtest_results()
        
        # Display last updated time
        st.markdown(f'<div class="last-update">Last updated: {self.last_updated.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
        
        if not backtest_results:
            st.info("No backtest results found.")
            return
        
        # Create backtest selector
        backtest_ids = [f"{b.get('strategy_id', 'Unknown')} - {b.get('symbol', 'Unknown')} ({b.get('timestamp', '').split('T')[0]})" for b in backtest_results]
        
        selected_backtest = st.selectbox(
            "Select Backtest Run",
            options=range(len(backtest_ids)),
            format_func=lambda x: backtest_ids[x]
        )
        
        backself.event_bus.request('data:live_feed') = backtest_results[selected_backtest]
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "💰 Equity Curve", 
            "📊 Trade Clusters", 
            "📈 Performance Metrics", 
            "🔍 Raw Data"
        ])
        
        with tab1:
            fig = self.create_equity_curve(backself.event_bus.request('data:live_feed'))
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not create equity curve visualization.")
            
            # Display key metrics
            if "performance_metrics" in backself.event_bus.request('data:live_feed'):
                metrics = backself.event_bus.request('data:live_feed')["performance_metrics"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", metrics.get("total_trades", 0))
                
                with col2:
                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                
                with col3:
                    st.metric("Net Profit", f"{metrics.get('net_profit', 0):.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        
        with tab2:
            fig = self.create_trade_cluster_viz(backself.event_bus.request('data:live_feed'))
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Could not create trade cluster visualization.")
        
        with tab3:
            metrics = self.create_performance_metrics(backself.event_bus.request('data:live_feed'))
            
            if metrics:
                # Display metrics in a grid
                col1, col2, col3 = st.columns(3)
                
                metrics_list = list(metrics.items())
                metrics_per_col = len(metrics_list) // 3 + (1 if len(metrics_list) % 3 > 0 else 0)
                
                for i, (key, value) in enumerate(metrics_list):
                    col_idx = i // metrics_per_col
                    
                    with [col1, col2, col3][col_idx]:
                        st.metric(key, value)
            else:
                st.info("No performance metrics available.")
        
        with tab4:
            # Show raw backtest data
            if "trades" in backself.event_bus.request('data:live_feed') and backself.event_bus.request('data:live_feed')["trades"]:
                st.markdown("### Backtest Trades")
                trades_df = pd.DataFrame(backself.event_bus.request('data:live_feed')["trades"])
                st.dataframe(trades_df, height=300)
            
            # Show backtest parameters
            if "parameters" in backself.event_bus.request('data:live_feed'):
                st.markdown("### Backtest Parameters")
                params_df = pd.DataFrame([backself.event_bus.request('data:live_feed')["parameters"]])
                st.dataframe(params_df, height=150)


# <!-- @GENESIS_MODULE_END: backtest_explorer -->