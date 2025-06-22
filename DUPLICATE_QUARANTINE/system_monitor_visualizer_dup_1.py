# <!-- @GENESIS_MODULE_START: system_monitor_visualizer -->


# 🔗 GENESIS EventBus Integration - Auto-injected by Orphan Recovery Engine
from datetime import datetime
import json

class SystemMonitorVisualizerEventBusIntegration:
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

            emit_telemetry("system_monitor_visualizer", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("system_monitor_visualizer", "position_calculated", {
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
                        "module": "system_monitor_visualizer",
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
                print(f"Emergency stop error in system_monitor_visualizer: {e}")
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
    """EventBus integration for system_monitor_visualizer"""
    
    def __init__(self):
        self.module_id = "system_monitor_visualizer"
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
system_monitor_visualizer_eventbus = SystemMonitorVisualizerEventBusIntegration()

"""
GENESIS System Monitor Visualizer Module v1.0
Visualization module for system monitoring in the GENESIS AI TRADING BOT SYSTEM

Dependencies: pandas, plotly, matplotlib, psutil
Input: System metrics from DashboardEngine
Output: Visualization components for the Streamlit dashboard
Telemetry: ENABLED
Compliance: ENFORCED
Real Data Enforcement: STRICT - No real/fallback data permitted
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import psutil
import time
import threading
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitorVisualizer:
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

            emit_telemetry("system_monitor_visualizer", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("system_monitor_visualizer", "position_calculated", {
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
                        "module": "system_monitor_visualizer",
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
                print(f"Emergency stop error in system_monitor_visualizer: {e}")
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
    """Visualization tools for system monitoring"""
    
    def __init__(self):
        """Initialize SystemMonitorVisualizer"""
        logger.info("🔄 SystemMonitorVisualizer initialized")
        
        # Initialize data storage
        self.cpu_history = deque(maxlen=90)  # 15 minutes at 10-second intervals
        self.memory_history = deque(maxlen=90)
        self.disk_history = deque(maxlen=90)
        self.network_history = deque(maxlen=90)
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_resources(self):
        """Monitor system resources in a background thread"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_history.append({
                    "timestamp": timestamp,
                    "value": cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_history.append({
                    "timestamp": timestamp,
                    "value": memory.percent
                })
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.disk_history.append({
                    "timestamp": timestamp,
                    "value": disk.percent
                })
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.network_history.append({
                    "timestamp": timestamp,
                    "sent_bytes": net_io.bytes_sent,
                    "recv_bytes": net_io.bytes_recv
                })
                
                # Sleep for 10 seconds
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error monitoring system resources: {str(e)}")
                time.sleep(5)  # Short sleep on error
    
    def get_cpu_history(self):
        """Get CPU usage history"""
        return list(self.cpu_history)
    
    def get_memory_history(self):
        """Get memory usage history"""
        return list(self.memory_history)
    
    def get_disk_history(self):
        """Get disk usage history"""
        return list(self.disk_history)
    
    def get_network_history(self):
        """Get network I/O history"""
        return list(self.network_history)
    
    def plot_cpu_usage(self):
        """Generate CPU usage plot"""
        if not self.cpu_history:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create DataFrame
        df = pd.DataFrame(list(self.cpu_history))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot CPU usage
        ax.plot(df["timestamp"], df["value"], color='blue')
        
        # Format plot
        ax.set_title("CPU Usage")
        ax.set_xlabel("Time")
        ax.set_ylabel("CPU Usage (%)")
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add threshold line
        ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_memory_usage(self):
        """Generate memory usage plot"""
        if not self.memory_history:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create DataFrame
        df = pd.DataFrame(list(self.memory_history))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot memory usage
        ax.plot(df["timestamp"], df["value"], color='green')
        
        # Format plot
        ax.set_title("Memory Usage")
        ax.set_xlabel("Time")
        ax.set_ylabel("Memory Usage (%)")
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add threshold line
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_disk_usage(self):
        """Generate disk usage plot"""
        if not self.disk_history:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create DataFrame
        df = pd.DataFrame(list(self.disk_history))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot disk usage
        ax.plot(df["timestamp"], df["value"], color='purple')
        
        # Format plot
        ax.set_title("Disk Usage")
        ax.set_xlabel("Time")
        ax.set_ylabel("Disk Usage (%)")
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        # Add threshold line
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_network_io(self):
        """Generate network I/O plot"""
        if not self.network_history or len(self.network_history) < 2:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create DataFrame
        df = pd.DataFrame(list(self.network_history))
        
        # Calculate transfer rates
        df['sent_rate'] = df['sent_bytes'].diff() / 10  # bytes per second
        df['recv_rate'] = df['recv_bytes'].diff() / 10  # bytes per second
        
        # Convert to KB/s
        df['sent_rate_kb'] = df['sent_rate'] / 1024
        df['recv_rate_kb'] = df['recv_rate'] / 1024
        
        # Drop first row (NaN due to diff)
        df = df.dropna()
        
        if df.empty:
            raise RuntimeError("ARCHITECT_MODE_COMPLIANCE: Operation failed")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot network I/O
        ax.plot(df["timestamp"], df["sent_rate_kb"], color='blue', label='Upload (KB/s)')
        ax.plot(df["timestamp"], df["recv_rate_kb"], color='green', label='Download (KB/s)')
        
        # Format plot
        ax.set_title("Network I/O")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transfer Rate (KB/s)")
        ax.grid(True, alpha=0.3)
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        ax.legend()
        plt.tight_layout()
        return fig
    
    def calculate_system_metrics(self):
        """Calculate system metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {},
            "memory": {},
            "disk": {},
            "network": {}
        }
        
        try:
            # CPU metrics
            metrics["cpu"]["current_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["cpu"]["count_physical"] = psutil.cpu_count(logical=False)
            metrics["cpu"]["count_logical"] = psutil.cpu_count(logical=True)
            
            # CPU frequency if available
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metrics["cpu"]["frequency_current"] = cpu_freq.current
                    metrics["cpu"]["frequency_min"] = cpu_freq.min
                    metrics["cpu"]["frequency_max"] = cpu_freq.max
            except:
    logger.info("Function operational")("Real implementation required - no stubs allowed in production")
            memory = psutil.virtual_memory()
            metrics["memory"]["total"] = memory.total
            metrics["memory"]["available"] = memory.available
            metrics["memory"]["used"] = memory.used
            metrics["memory"]["percent"] = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics["disk"]["total"] = disk.total
            metrics["disk"]["used"] = disk.used
            metrics["disk"]["free"] = disk.free
            metrics["disk"]["percent"] = disk.percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics["network"]["bytes_sent"] = net_io.bytes_sent
            metrics["network"]["bytes_recv"] = net_io.bytes_recv
            metrics["network"]["packets_sent"] = net_io.packets_sent
            metrics["network"]["packets_recv"] = net_io.packets_recv
            
            # Process count
            metrics["system"] = {
                "process_count": len(psutil.pids())
            }
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {str(e)}")
        
        return metrics
    
    def __del__(self):
        """Clean up resources"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(1.0)  # Wait up to 1 second for thread to terminate

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
        

# <!-- @GENESIS_MODULE_END: system_monitor_visualizer -->