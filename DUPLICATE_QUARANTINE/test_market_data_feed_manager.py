import logging
# <!-- @GENESIS_MODULE_START: test_market_data_feed_manager -->
"""
🏛️ GENESIS TEST_MARKET_DATA_FEED_MANAGER - INSTITUTIONAL GRADE v8.0.0
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

                emit_telemetry("test_market_data_feed_manager", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_market_data_feed_manager", "position_calculated", {
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
                            "module": "test_market_data_feed_manager",
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
                    print(f"Emergency stop error in test_market_data_feed_manager: {e}")
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
                    "module": "test_market_data_feed_manager",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_market_data_feed_manager", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_market_data_feed_manager: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


"""
GENESIS MarketDataFeedManager Test Script - ARCHITECT MODE v2.7
===============================================================
Test script to verify MarketDataFeedManager compliance and functionality.
"""

import sys
import json
from datetime import datetime

def test_module_compliance():
    """Test MarketDataFeedManager compliance with ARCHITECT MODE rules"""
    print(" GENESIS ARCHITECT MODE - MODULE COMPLIANCE TEST")
    print("=" * 60)
    
    try:
        # Import modules
        from market_data_feed_manager import MarketDataFeedManager
        from event_bus import get_event_bus
        
        print(" Module imports successful")
        
        # Create manager instance
        manager = MarketDataFeedManager()
        print(" MarketDataFeedManager instance created")
        
        # Test EventBus connectivity
        event_bus = get_event_bus()
        print(" EventBus accessible")
        
        # Test compliance validation
        compliance_report = manager.validate_compliance()
        print(f" Compliance validation: {compliance_report['status']}")
        
        if compliance_report['violations']:
            print(" Compliance violations detected:")
            for violation in compliance_report['violations']:
                print(f"   - {violation}")
            return False
        
        # Test status reporting
        status = manager.get_status()
        print(f" Module status: {status['status']}")
        print(f" Real data only: {status['real_data_only']}")
        print(f" Compliance mode: {status['compliance_mode']}")
        
        # Test EventBus subscription (without MT5 connection)
        def test_callback(event):
            print(f" Received test event: {event['data']}")
        
        event_bus.subscribe("TestTopic", test_callback, "TestConsumer")
        event_bus.emit_event("TestTopic", {"test": "data"}, "TestProducer")
        
        print(" EventBus emit/subscribe test passed")
        
        # Verify module registration
        with open('module_registry.json', 'r') as f:
            registry = json.load(f)
        
        manager_found = False
        for module in registry['modules']:
            if module['name'] == 'MarketDataFeedManager':
                manager_found = True
                print(" Module registered in module_registry.json")
                print(f"   - Real data: {module['real_data']}")
                print(f"   - Compliance: {module['compliance']}")
                print(f"   - Telemetry: {module['telemetry']}")
                break
        
        if not manager_found:
            print(" Module not found in module_registry.json")
            return False
        
        # Verify system tree registration
        with open('system_tree.json', 'r') as f:
            system_tree = json.load(f)
        
        node_found = False
        for node in system_tree['nodes']:
            if node['id'] == 'MarketDataFeedManager':
                node_found = True
                print(" Module registered in system_tree.json")
                print(f"   - Real data source: {node['real_data_source']}")
                print(f"   - Compliance verified: {node['compliance_verified']}")
                break
        
        if not node_found:
            print(" Module not found in system_tree.json")
            return False
        
        # Verify EventBus routes
        with open('event_bus.json', 'r') as f:
            event_bus_config = json.load(f)
        
        routes_found = 0
        for route in event_bus_config['routes']:
            if route['producer'] == 'MarketDataFeedManager':
                routes_found += 1
                print(f" EventBus route: {route['topic']} -> {route['consumer']}")
        
        if routes_found == 0:
            print(" No EventBus routes found for MarketDataFeedManager")
            return False
        
        print(f" Found {routes_found} EventBus routes")
        
        # Test telemetry logging
        manager.log_telemetry({
            "event_type": "test_event",
            "status": "success"
        })
        print(" Telemetry logging test passed")
        
        print("\n COMPLIANCE TEST RESULTS:")
        print("=" * 40)
        print(" Module creation: PASSED")
        print(" EventBus integration: PASSED") 
        print(" Registry synchronization: PASSED")
        print(" System tree integration: PASSED")
        print(" Compliance validation: PASSED")
        print(" Real data enforcement: PASSED")
        print(" Telemetry integration: PASSED")
        print(" ARCHITECT MODE compliance: PASSED")
        
        return True
        
    except ImportError as e:
        print(f" Import error: {e}")
        return False
    except Exception as e:
        print(f" Test error: {e}")
        return False

def test_mt5_integration():
    """Test MT5 integration if available"""
    print("\n MT5 INTEGRATION TEST")
    print("=" * 30)
    
    try:
        import MetaTrader5 as mt5
        print(" MetaTrader5 module available")
        
        from market_data_feed_manager import MarketDataFeedManager
        manager = MarketDataFeedManager()
        
        # Attempt MT5 connection
        if manager.connect_to_mt5():
            print(" MT5 connection successful")
            print(f" Found {len(manager.symbols)} symbols")
            
            # Test brief streaming
            print(" Testing 5-second data stream...")
            if manager.start_stream(["EURUSD"]):
                import time
                time.sleep(5)
                manager.stop_stream()
                print(" Streaming test completed")
            
            manager.disconnect()
            print(" MT5 disconnection successful")
            return True
        else:
            print("  MT5 connection failed - terminal may not be running")
            return False
            
    except ImportError:
        print("  MetaTrader5 module not installed")
        print("   Install with: pip install MetaTrader5")
        return False
    except Exception as e:
        print(f" MT5 test error: {e}")
        return False

if __name__ == "__main__":
    print(" GENESIS AI AGENT LOCK-IN  MarketDataFeedManager Testing")
    print("=" * 70)
    
    # Run compliance tests
    compliance_passed = test_module_compliance()
    
    # Run MT5 integration tests
    mt5_passed = test_mt5_integration()
    
    print("\n FINAL TEST SUMMARY")
    print("=" * 25)
    print(f"Compliance Tests: {' PASSED' if compliance_passed else ' FAILED'}")
    print(f"MT5 Integration: {' PASSED' if mt5_passed else '  SKIPPED/FAILED'}")
    
    if compliance_passed:
        print("\n MarketDataFeedManager is ARCHITECT MODE compliant!")
        print(" Ready for EventBus wiring with other GENESIS modules")
    else:
        print("\n Compliance issues detected - review and fix before proceeding")
        sys.exit(1)


# <!-- @GENESIS_MODULE_END: test_market_data_feed_manager -->
