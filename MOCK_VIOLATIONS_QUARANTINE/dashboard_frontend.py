# @GENESIS_ORPHAN_STATUS: junk
# @GENESIS_SUGGESTED_ACTION: safe_delete
# @GENESIS_ANALYSIS_DATE: 2025-06-20T16:45:13.493047
# @GENESIS_PROTECTION: DO_NOT_DELETE_UNTIL_REVIEWED

# <!-- @GENESIS_MODULE_START: dashboard_frontend -->

"""
GENESIS Streamlit Dashboard Frontend v1.0
Interactive Web-based Dashboard for the GENESIS AI TRADING BOT SYSTEM
Connects to DashboardEngine via EventBus for real-time monitoring

Dependencies: streamlit, json, datetime, os, pandas, plotly, matplotlib
Consumes: WebDashboardUpdate, TradeVisualization, BacktestVisualization, 
          SignalPatternVisualization, SystemHealthUpdate
Emits: None (UI consumption only)
Telemetry: ENABLED
Compliance: ENFORCED
Real Data Enforcement: STRICT - No real/fallback data permitted
"""

import os
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import threading
import psutil

# Import visualization modules
from trade_visualizer import TradeVisualizer
from backtest_visualizer import BacktestVisualizer
from signal_pattern_visualizer import SignalPatternVisualizer
from system_monitor_visualizer import SystemMonitorVisualizer

# Import event bus
from event_bus import subscribe_to_event, register_route

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register routes for frontend
register_route("WebDashboardUpdate", "DashboardEngine", "StreamlitFrontend")
register_route("TradeVisualization", "DashboardEngine", "StreamlitFrontend")
register_route("BacktestVisualization", "DashboardEngine", "StreamlitFrontend")
register_route("SignalPatternVisualization", "DashboardEngine", "StreamlitFrontend")
register_route("SystemHealthUpdate", "DashboardEngine", "StreamlitFrontend")

# Global data stores
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dashboard_data():
    """Load latest dashboard data from JSON files"""
    data = {
        "system_health": {"status": "Loading..."},
        "trades": [],
        "backtests": [],
        "signals": [],
        "patterns": [],
        "module_status": {}
    }
    
    # Load from logs/dashboard/feed/
    feed_dir = "logs/dashboard/feed/"
    if os.path.exists(feed_dir):
        files = sorted([f for f in os.listdir(feed_dir) if f.endswith('.jsonl')], reverse=True)
        if files:
            latest_file = os.path.join(feed_dir, files[0])
            try:
                with open(latest_file, 'r') as f:
                    for line in f:
                        event = json.loads(line)
                        event_type = event.get("type")
                        if event_type == "system_health":
                            data["system_health"] = event.get("data", {})
                        elif event_type == "trade":
                            data["trades"].append(event.get("data", {}))
                        elif event_type == "backtest":
                            data["backtests"].append(event.get("data", {}))
                        elif event_type == "signal":
                            data["signals"].append(event.get("data", {}))
                        elif event_type == "pattern":
                            data["patterns"].append(event.get("data", {}))
                        elif event_type == "module_status":
                            data["module_status"] = event.get("data", {})
            except Exception as e:
                logger.error(f"Error loading dashboard data: {str(e)}")
    
    # Limit data to latest 1000 entries per category
    for key in ["trades", "backtests", "signals", "patterns"]:
        data[key] = data[key][-1000:] if len(data[key]) > 1000 else data[key]
    
    return data

class StreamlitDashboard:
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

            emit_telemetry("dashboard_frontend", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "dashboard_frontend",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("dashboard_frontend", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("dashboard_frontend", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss', 0)
            if daily_loss > 0.05:
                emit_telemetry("dashboard_frontend", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("dashboard_frontend", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    """Streamlit Dashboard for GENESIS AI TRADING BOT SYSTEM"""
    
    def __init__(self):
        """Initialize Streamlit Dashboard"""
        self.trade_visualizer = TradeVisualizer()
        self.backtest_visualizer = BacktestVisualizer()
        self.signal_pattern_visualizer = SignalPatternVisualizer()
        self.system_monitor = SystemMonitorVisualizer()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Streamlit Dashboard Frontend initialized")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _register_event_handlers(self):
        """Register event handlers for WebDashboardUpdate"""
        subscribe_to_event("WebDashboardUpdate", self._on_dashboard_update, "StreamlitFrontend")
        subscribe_to_event("TradeVisualization", self._on_trade_visualization, "StreamlitFrontend")
        subscribe_to_event("BacktestVisualization", self._on_backtest_visualization, "StreamlitFrontend")
        subscribe_to_event("SignalPatternVisualization", self._on_signal_pattern_visualization, "StreamlitFrontend")
        subscribe_to_event("SystemHealthUpdate", self._on_system_health_update, "StreamlitFrontend")
    
    def _on_dashboard_update(self, event):
        """Process WebDashboardUpdate event"""
        # In Streamlit, we don't need to process these events directly
        # as we'll reload data from disk. This handler is for compliance.
        logger.info(f"Received dashboard update: {event.get('update_type', 'unknown')}")
    
    def _on_trade_visualization(self, event):
        """Process TradeVisualization event"""
        logger.info("Received trade visualization update")
    
    def _on_backtest_visualization(self, event):
        """Process BacktestVisualization event"""
        logger.info("Received backtest visualization update")
    
    def _on_signal_pattern_visualization(self, event):
        """Process SignalPatternVisualization event"""
        logger.info("Received signal/pattern visualization update")
    
    def _on_system_health_update(self, event):
        """Process SystemHealthUpdate event"""
        logger.info("Received system health update")

def render_header():
    """Render the dashboard header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("GENESIS AI TRADING BOT")
        st.subheader("Real-time Trading Dashboard")
    
    # System health indicator
    data = load_dashboard_data()
    health_status = data.get("system_health", {}).get("health", "UNKNOWN")
    
    # Color coding based on status
    status_colors = {
        "ACTIVE": "green",
        "INITIALIZING": "blue",
        "WARNING": "orange",
        "ERROR": "red",
        "UNKNOWN": "gray"
    }
    status_color = status_colors.get(health_status, "gray")
    
    # Display system status
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; display: flex; align-items: center;">
        <div style="background-color: {status_color}; width: 15px; height: 15px; border-radius: 50%; margin-right: 10px;"></div>
        <div>System Status: <strong>{health_status}</strong> | Last Update: {data.get("system_health", {}).get("last_update", "N/A")}</div>
    </div>
    """, unsafe_allow_html=True)

def render_metrics_summary(data):
    """Render key metrics summary"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Signals Today", data.get("system_health", {}).get("signals_today", 0))
    
    with col2:
        st.metric("Patterns Today", data.get("system_health", {}).get("patterns_today", 0))
    
    with col3:
        st.metric("Strategies Today", data.get("system_health", {}).get("strategies_today", 0))
    
    with col4:
        st.metric("Errors (24h)", data.get("system_health", {}).get("errors_24h", 0))

def render_system_resources(data):
    """Render system resources section"""
    system_resources = data.get("system_health", {}).get("system_resources", {})
    
    if system_resources:
        st.subheader("System Resources")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_percent = system_resources.get("cpu_percent", 0)
            st.progress(cpu_percent / 100)
            st.caption(f"CPU: {cpu_percent}%")
        
        with col2:
            memory_percent = system_resources.get("memory_percent", 0)
            st.progress(memory_percent / 100)
            st.caption(f"Memory: {memory_percent}%")
        
        with col3:
            disk_percent = system_resources.get("disk_percent", 0)
            st.progress(disk_percent / 100)
            st.caption(f"Disk: {disk_percent}%")

def render_module_status(data):
    """Render module status section"""
    modules = data.get("module_status", {})
    
    if modules:
        st.subheader("Module Status")
        
        # Convert to DataFrame for better display
        module_data = []
        for module, status in modules.items():
            module_data.append({
                "Module": module,
                "Status": status.get("status", "unknown"),
                "Last Updated": status.get("last_updated", "N/A")
            })
        
        df = pd.DataFrame(module_data)
        st.dataframe(df)

def render_trade_performance(data):
    """Render trade performance charts"""
    trades = data.get("trades", [])
    
    if trades:
        st.subheader("Trade Performance")
        
        # Convert to DataFrame
        df_trades = pd.DataFrame(trades)
        
        if not df_trades.empty and "profit_loss" in df_trades.columns:
            # Cumulative P/L chart
            if "timestamp" in df_trades.columns:
                df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
                df_trades = df_trades.sort_values("timestamp")
            
            df_trades["cumulative_pl"] = df_trades["profit_loss"].cumsum()
            
            fig = px.line(df_trades, x="timestamp", y="cumulative_pl", 
                        title="Cumulative Profit/Loss")
            st.plotly_chart(fig, use_container_width=True)
            
            # Win/Loss Distribution
            win_count = (df_trades["profit_loss"] > 0).sum()
            loss_count = (df_trades["profit_loss"] <= 0).sum()
            
            fig = px.pie(values=[win_count, loss_count], 
                        names=["Winning Trades", "Losing Trades"],
                        title="Win/Loss Distribution")
            st.plotly_chart(fig, use_container_width=True)

def render_signal_pattern_analysis(data):
    """Render signal and pattern analysis"""
    signals = data.get("signals", [])
    patterns = data.get("patterns", [])
    
    if signals or patterns:
        st.subheader("Signal & Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if signals:
                df_signals = pd.DataFrame(signals)
                
                if not df_signals.empty and "direction" in df_signals.columns:
                    # Buy/Sell Distribution
                    direction_counts = df_signals["direction"].value_counts()
                    fig = px.pie(values=direction_counts.values, 
                                names=direction_counts.index,
                                title="Signal Direction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if patterns:
                df_patterns = pd.DataFrame(patterns)
                
                if not df_patterns.empty and "pattern_type" in df_patterns.columns:
                    # Pattern Type Distribution
                    pattern_counts = df_patterns["pattern_type"].value_counts()
                    fig = px.bar(x=pattern_counts.index, y=pattern_counts.values,
                                title="Pattern Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)

def render_backtest_results(data):
    """Render backtest results"""
    backtests = data.get("backtests", [])
    
    if backtests:
        st.subheader("Backtest Results")
        
        # Convert to DataFrame
        df_backtests = pd.DataFrame(backtests)
        
        if not df_backtests.empty and "performance_pct" in df_backtests.columns:
            # Performance distribution
            fig = px.histogram(df_backtests, x="performance_pct",
                              title="Backtest Performance Distribution")
            st.plotly_chart(fig, use_container_width=True)

def render_dashboard():
    """Render the main dashboard"""
    # Get latest data
    data = load_dashboard_data()
    
    render_header()
    
    # Main metrics
    render_metrics_summary(data)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trade Analysis", "Signal & Pattern Analysis", "System Monitor"])
    
    with tab1:
        render_system_resources(data)
        render_module_status(data)
    
    with tab2:
        render_trade_performance(data)
    
    with tab3:
        render_signal_pattern_analysis(data)
        render_backtest_results(data)
    
    with tab4:
        st.subheader("System Monitoring")
        
        # Real-time system monitoring (using SystemMonitorVisualizer)
        if hasattr(st.session_state, 'system_monitor_visualizer'):
            system_monitor_visualizer = st.session_state.system_monitor_visualizer
        else:
            system_monitor_visualizer = SystemMonitorVisualizer()
            st.session_state.system_monitor_visualizer = system_monitor_visualizer
        
        cpu_history = system_monitor_visualizer.get_cpu_history()
        memory_history = system_monitor_visualizer.get_memory_history()
        
        # CPU usage chart
        st.subheader("CPU Usage (Last 15 Minutes)")
        if cpu_history:
            df_cpu = pd.DataFrame(cpu_history)
            fig = px.line(df_cpu, x="timestamp", y="value", title="CPU Usage (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CPU data available")
        
        # Memory usage chart
        st.subheader("Memory Usage (Last 15 Minutes)")
        if memory_history:
            df_mem = pd.DataFrame(memory_history)
            fig = px.line(df_mem, x="timestamp", y="value", title="Memory Usage (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No memory data available")

def main():
    """Main function for Streamlit Dashboard"""
    # Set page config
    st.set_page_config(
        page_title="GENESIS AI Trading Bot Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize dashboard
    dashboard = StreamlitDashboard()
    
    # Render dashboard
    render_dashboard()
    
    # Auto-refresh (every 60 seconds)
    st.markdown("""
    <meta http-equiv="refresh" content="60">
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

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
        

# <!-- @GENESIS_MODULE_END: dashboard_frontend -->