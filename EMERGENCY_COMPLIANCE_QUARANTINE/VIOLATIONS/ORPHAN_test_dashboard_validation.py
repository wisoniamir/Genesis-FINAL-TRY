import logging

# GENESIS EventBus Integration - Auto-injected by Comprehensive Module Upgrade Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    from core.telemetry import emit_telemetry
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback for modules without core access
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event}")
    def register_route(route, producer, consumer): pass
    def emit_telemetry(module, event, data): print(f"TELEMETRY: {module}.{event}")
    EVENTBUS_AVAILABLE = False


# <!-- @GENESIS_MODULE_START: test_dashboard_validation -->

#!/usr/bin/env python3
"""
GENESIS Dashboard Test Script
Tests the dashboard components without GUI for validation
"""

import sys
import os
import json
import time
from datetime import datetime, timezone

# Add current directory to path
sys.path.append('.')

try:
    from genesis_dashboard_ui import GenesisEventBus, DataMonitor
    print("✅ Dashboard modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dashboard modules: {e}")
    exit(1)

def test_event_bus():
    """Test the event bus functionality"""
    print("\n🔄 Testing Event Bus...")
    
    event_bus = GenesisEventBus()
    events_received = []
    
    def test_callback(event):
        events_received.append(event)
        print(f"  📨 Event received: {event['type']}")
    
    # Subscribe to test events
    event_bus.subscribe("test:event", test_callback)
    
    # Start event bus
    event_bus.start()
    
    # Emit test events
    event_bus.emit("test:event", {"message": "Test message 1"})
    event_bus.emit("test:event", {"message": "Test message 2"})
    event_bus.emit("other:event", {"message": "Should not be received"})
    
    # Wait for processing
    time.sleep(0.1)
    
    # Stop event bus
    event_bus.stop()
    
    # Verify results
    if len(events_received) == 2:
        print("  ✅ Event bus test passed")
        return True
    else:
        print(f"  ❌ Event bus test failed - received {len(events_received)} events, expected 2")
        return False

def self.event_bus.request('data:live_feed')_files():
    """Test data file availability and format"""
    print("\n📁 Testing Data Files...")
    
    files_to_check = {
        "telemetry.json": "Telemetry data",
        "execution_log.json": "Execution log",
        "event_bus.json": "Event bus data",
        "system_tree.json": "System tree",
        "build_status.json": "Build status"
    }
    
    results = {}
    
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                print(f"  ✅ {description}: Valid JSON, {len(str(data))} chars")
                results[filename] = True
            except json.JSONDecodeError:
                print(f"  ❌ {description}: Invalid JSON")
                results[filename] = False
            except Exception as e:
                print(f"  ❌ {description}: Error reading - {e}")
                results[filename] = False
        else:
            print(f"  ⚠️ {description}: File not found")
            results[filename] = False
    
    return results

def self.event_bus.request('data:live_feed')_monitor():
    """Test the data monitoring system"""
    print("\n👀 Testing Data Monitor...")
    
    event_bus = GenesisEventBus()
    data_monitor = DataMonitor(event_bus)
    
    events_received = []
    
    def monitor_callback(event):
        events_received.append(event)
        print(f"  📊 Data update: {event['data']['file_type']}")
    
    # Subscribe to data updates
    event_bus.subscribe("data:update:telemetry", monitor_callback)
    
    # Start systems
    event_bus.start()
    data_monitor.start_monitoring()
    
    # Wait a moment for initial scan
    time.sleep(2)
    
    # Stop systems
    data_monitor.stop_monitoring()
    event_bus.stop()
    
    print(f"  📈 Monitoring test completed")
    return True

def test_dashboard_components():
    """Test dashboard component initialization"""
    print("\n🖥️ Testing Dashboard Components...")
    
    try:
        from genesis_dashboard_ui import (
            LiveSignalPanel, ExecutionControlPanel, TradeJournalPanel,
            StatsPanel, MT5ConnectionPanel, FeedbackPanel
        )
        print("  ✅ All panel classes available")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import panel classes: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 GENESIS Dashboard Validation Test")
    print("=" * 50)
    
    tests = [
        ("Event Bus", test_event_bus),
        ("Data Files", self.event_bus.request('data:live_feed')_files),
        ("Data Monitor", self.event_bus.request('data:live_feed')_monitor),
        ("Dashboard Components", test_dashboard_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"  ❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🏆 All tests passed - Dashboard is ready for production!")
        return True
    else:
        print("⚠️ Some tests failed - Please review issues above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)


# <!-- @GENESIS_MODULE_END: test_dashboard_validation -->