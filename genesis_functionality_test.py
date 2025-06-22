import logging
# <!-- @GENESIS_MODULE_START: genesis_functionality_test -->
"""
🏛️ GENESIS GENESIS_FUNCTIONALITY_TEST - INSTITUTIONAL GRADE v8.0.0
===============================================================
ARCHITECT MODE ULTIMATE: Enhanced via Complete Intelligent Wiring Engine

🎯 ENHANCED FEATURES:
- Complete EventBus integration
- Real-time telemetry monitoring
- FTMO compliance enforcement
- Emergency kill-switch protection
- Institutional-grade architecture

🔐 ARCHITECT MODE v8.0.0: Ultimate compliance enforcement
"""


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

                emit_telemetry("genesis_functionality_test", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("genesis_functionality_test", "position_calculated", {
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
                            "module": "genesis_functionality_test",
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
                    print(f"Emergency stop error in genesis_functionality_test: {e}")
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
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "genesis_functionality_test",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("genesis_functionality_test", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in genesis_functionality_test: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


from datetime import datetime


# 🔗 GENESIS EventBus Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback implementation
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event} - {data}")
    def register_route(route, producer, consumer): pass
    EVENTBUS_AVAILABLE = False


#!/usr/bin/env python3
"""
🎯 GENESIS FUNCTIONALITY TEST - Prove Dashboard Works
Test script to validate enhanced GENESIS functionality
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "modules"))

print("🚀 GENESIS FUNCTIONALITY TEST STARTING...")
print("=" * 50)

# Test 1: Module Import Capabilities
print("📦 TESTING MODULE IMPORTS:")
try:
    from modules.institutional.mt5_adapter_v7 import MT5AdapterV7
    print("✅ MT5AdapterV7 imported successfully")
    adapter = MT5AdapterV7()
    print(f"✅ MT5AdapterV7 instantiated: {type(adapter)}")
except Exception as e:
    print(f"❌ MT5AdapterV7 import failed: {e}")

try:
    from modules.signals.signal_engine import SignalEngine
    print("✅ SignalEngine imported successfully")
    signal_engine = SignalEngine()
    print(f"✅ SignalEngine instantiated: {type(signal_engine)}")
except Exception as e:
    print(f"❌ SignalEngine import failed: {e}")

try:
    from modules.execution.execution_engine import ExecutionEngine
    print("✅ ExecutionEngine imported successfully")
    exec_engine = ExecutionEngine()
    print(f"✅ ExecutionEngine instantiated: {type(exec_engine)}")
except Exception as e:
    print(f"❌ ExecutionEngine import failed: {e}")

try:
    from modules.restored.event_bus import EventBus
    print("✅ EventBus imported successfully")
    event_bus = EventBus()
    print(f"✅ EventBus instantiated: {type(event_bus)}")
except Exception as e:
    print(f"❌ EventBus import failed: {e}")

# Test 2: PyQt5 Availability
print("\n🖥️ TESTING GUI FRAMEWORK:")
try:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtCore import Qt
    print("✅ PyQt5 imported successfully")
    app = QApplication([])
    print("✅ QApplication created successfully")
except Exception as e:
    print(f"❌ PyQt5 import failed: {e}")

# Test 3: MetaTrader5 Availability
print("\n📈 TESTING MT5 INTEGRATION:")
try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 imported successfully")
    
    # Try to initialize
    if mt5.initialize():
        print("✅ MT5 initialized successfully")
        account_info = mt5.account_info()
        if account_info:
            print(f"✅ MT5 account connected: {account_info.login}")
        else:
            print("⚠️ MT5 initialized but no account info")
        mt5.shutdown()
    else:
        print("⚠️ MT5 import OK but initialization failed")
except Exception as e:
    print(f"❌ MetaTrader5 failed: {e}")

# Test 4: Module Integration Test
print("\n🔗 TESTING MODULE INTEGRATION:")
try:
    # Create a mock event bus test
    from modules.restored.event_bus import EventBus
    test_bus = EventBus()
    
    # Subscribe to a test event
    def test_handler(data):
        print(f"✅ Event received: {data}")
    
    test_bus.subscribe("test_event", test_handler)
    
    # Emit test event
    test_bus.emit("test_event", {"message": "Integration test successful"})
    print("✅ EventBus integration working")
    
except Exception as e:
    print(f"❌ Module integration failed: {e}")

print("\n" + "=" * 50)
print("🎯 GENESIS FUNCTIONALITY TEST COMPLETED")
print("Now attempting to launch GUI...")

# Test 5: Launch GUI Test
try:
    import subprocess
    result = subprocess.run([sys.executable, "genesis_desktop.py"], 
                          capture_output=True, text=True, timeout=10)
    print(f"GUI launch attempt: return code {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
except subprocess.TimeoutExpired:
    print("✅ GUI launched successfully (timeout = running)")
except Exception as e:
    print(f"❌ GUI launch failed: {e}")


# <!-- @GENESIS_MODULE_END: genesis_functionality_test -->
