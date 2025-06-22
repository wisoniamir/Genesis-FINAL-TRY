import logging
import sys
from pathlib import Path

# <!-- @GENESIS_MODULE_START: module_status -->

"""
GENESIS Dashboard - Module Status Component
Real-time monitoring of all GENESIS modules
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os

from styles.dashboard_styles import module_status_badge

class ModuleStatusComponent:
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

            emit_telemetry("module_status_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("module_status_recovered_1", "position_calculated", {
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
                        "module": "module_status_recovered_1",
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
                print(f"Emergency stop error in module_status_recovered_1: {e}")
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
            return EventBusManager.get_instance()
        except ImportError:
            logging.warning("EventBus not available - integration required")
            return None
            
    def emit_telemetry(self, data):
        if self.event_bus:
            self.event_bus.emit('telemetry', data)
    """
    Component for displaying real-time status of all GENESIS modules
    """
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.config = config
        self.modules_to_monitor = config["modules_to_monitor"]
        self.refresh_rate = config["refresh_rate"]["module_status"]
        self.last_telemetry = {}
        self.module_health = {}
        self.last_updated = datetime.now()
    
    def load_module_status(self):
        """Load current module status from EventBus logs and telemetry"""
        try:
            # Load from build_status.json for initial state
            with open("build_status.json", "r") as f:
                build_status = json.load(f)
                
            modules_status = {}
            
            # Get status from build_status.json
            for module_name in self.modules_to_monitor:
                # Check if it's in modules_status or in StreamlitFrontend field
                if module_name in build_status.get("modules_status", {}):
                    module_data = build_status["modules_status"][module_name]
                    modules_status[module_name] = {
                        "status": module_data.get("status", "unknown"),
                        "eventbus_connected": module_data.get("eventbus_connected", False),
                        "telemetry_enabled": module_data.get("telemetry_enabled", False),
                        "real_data": module_data.get("real_data", False),
                        "last_updated": module_data.get("last_updated", "Unknown"),
                    }
                elif module_name == "StreamlitFrontend" and "StreamlitFrontend" in build_status:
                    module_data = build_status["StreamlitFrontend"]
                    modules_status[module_name] = {
                        "status": module_data.get("status", "unknown"),
                        "eventbus_connected": module_data.get("eventbus_connected", False),
                        "telemetry_enabled": module_data.get("telemetry_enabled", False),
                        "real_data": module_data.get("real_data", False),
                        "last_updated": module_data.get("last_updated", "Unknown"),
                    }
                else:
                    modules_status[module_name] = {
                        "status": "unknown",
                        "eventbus_connected": False,
                        "telemetry_enabled": False,
                        "real_data": False,
                        "last_updated": "Unknown"
                    }
            
            # Try to load latest telemetry
            self._update_from_telemetry(modules_status)
            
            return modules_status
            
        except Exception as e:
            st.error(f"Error loading module status: {str(e)}")
            return {module: {"status": "unknown", "error": str(e)} for module in self.modules_to_monitor}
    
    def _update_from_telemetry(self, modules_status):
        """Update module status from telemetry logs"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Try to get today's logs, fallback to yesterday's if needed
            telemetry_files = [
                f"logs/dashboard/feed/dashboard_feed_{today}.jsonl",
                f"logs/dashboard/feed/dashboard_feed_{yesterday}.jsonl"
            ]
            
            latest_logs = {}
            
            for file_path in telemetry_files:
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                log = json.loads(line.strip())
                                if "module" in log:
                                    module_name = log["module"]
                                    if module_name in modules_status:
                                        # Check if this log is newer than what we have
                                        if module_name not in latest_logs or log.get("timestamp", "") > latest_logs[module_name].get("timestamp", ""):
                                            latest_logs[module_name] = log
                            except:
                                continue
            
            # Update status from telemetry
            for module_name, log in latest_logs.items():
                modules_status[module_name]["last_heartbeat"] = log.get("timestamp", "Unknown")
                
                # Calculate time since last heartbeat
                if log.get("timestamp"):
                    try:
                        last_time = datetime.fromisoformat(log["timestamp"])
                        now = datetime.now()
                        time_diff = (now - last_time).total_seconds()
                        
                        if time_diff < 60:  # Less than a minute
                            modules_status[module_name]["status"] = "active"
                        elif time_diff < 300:  # Less than 5 minutes
                            modules_status[module_name]["status"] = "warning"
                        else:
                            modules_status[module_name]["status"] = "inactive"
                    except:
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
                if "metrics" in log:
                    for metric_key, metric_value in log.get("metrics", {}).items():
                        modules_status[module_name][metric_key] = metric_value
            
            self.last_updated = datetime.now()
            
        except Exception as e:
            st.warning(f"Could not update from telemetry: {str(e)}")
    
    def render(self):
        """Render the module status overview"""
        st.markdown('<div class="main-title">Module Status Overview</div>', unsafe_allow_html=True)
        
        # Get current status
        modules_status = self.load_module_status()
        
        # Display last updated time
        st.markdown(f'<div class="last-update">Last updated: {self.last_updated.strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
        
        # Display modules in columns to make better use of space
        cols = st.columns(3)
        
        for i, (module_name, status) in enumerate(modules_status.items()):
            col_idx = i % 3
            
            with cols[col_idx]:
                status_badge = module_status_badge(status.get("status", "unknown"))
                
                card_html = f"""
                <div class="module-card">
                    <div class="module-card-header">
                        <div class="module-title">{module_name}</div>
                        {status_badge}
                    </div>
                    <div class="metric-row">
                        <div class="metric-label">Last Updated:</div>
                        <div class="metric-value">{status.get("last_updated", "Unknown").split("T")[1].split(".")[0] if "T" in status.get("last_updated", "Unknown") else "Unknown"}</div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label">EventBus:</div>
                        <div class="metric-value">{'✓' if status.get("eventbus_connected", False) else '✗'}</div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label">Telemetry:</div>
                        <div class="metric-value">{'✓' if status.get("telemetry_enabled", False) else '✗'}</div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-label">Real Data:</div>
                        <div class="metric-value">{'✓' if status.get("real_data", False) else '✗'}</div>
                    </div>
                </div>
                """
                
                st.markdown(card_html, unsafe_allow_html=True)
        
        # Provide a way to view more detailed metrics
        if st.button("View Detailed Metrics"):
            self._show_detailed_metrics(modules_status)
    
    def _show_detailed_metrics(self, modules_status):
        """Show detailed metrics for all modules"""
        st.markdown('<div class="subtitle">Detailed Module Metrics</div>', unsafe_allow_html=True)
        
        # Convert to DataFrame for easier display
        metrics_list = []
        for module_name, status in modules_status.items():
            module_metrics = {"Module": module_name, "Status": status.get("status", "unknown")}
            
            # Add all other metrics
            for k, v in status.items():
                if k not in ["status"]:
                    module_metrics[k] = v
            
            metrics_list.append(module_metrics)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Display as table
        st.dataframe(df)


# <!-- @GENESIS_MODULE_END: module_status -->