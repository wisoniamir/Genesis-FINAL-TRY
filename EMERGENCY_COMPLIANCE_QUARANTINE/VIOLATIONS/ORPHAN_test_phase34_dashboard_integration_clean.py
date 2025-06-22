# <!-- @GENESIS_MODULE_START: test_phase34_dashboard_integration_clean -->

#!/usr/bin/env python3
"""
GENESIS Phase 34 - Dashboard Integration Validation Test
Validates the complete Phase 34 broker discovery and telemetry dashboard integration
ARCHITECT MODE v2.9 - STRICT COMPLIANCE
"""

import json
import logging
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_components.broker_discovery_panel import BrokerDiscoveryPanel

class Phase34DashboardIntegrationTest:
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

            emit_telemetry("ORPHAN_test_phase34_dashboard_integration_clean", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "ORPHAN_test_phase34_dashboard_integration_clean",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("ORPHAN_test_phase34_dashboard_integration_clean", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("ORPHAN_test_phase34_dashboard_integration_clean", "position_calculated", {
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
                emit_telemetry("ORPHAN_test_phase34_dashboard_integration_clean", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("ORPHAN_test_phase34_dashboard_integration_clean", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
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
    """Phase 34 Dashboard Integration Test Suite"""
    
    def __init__(self):
        self.test_results = {
            "start_time": datetime.utcnow().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # Setup logger
        self.logger = logging.getLogger("Phase34DashboardIntegrationTest")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("Starting Phase 34 Dashboard Integration Test Suite")
    
    def log_test_result(self, test_name, passed, details):
        """Log test result and update tracking"""
        status = "PASS" if passed else "FAIL"
        self.logger.info(f"{status}: {test_name} - {details}")
        
        self.test_results["tests_run"] += 1
        if passed:
            self.test_results["tests_passed"] += 1
        else:
            self.test_results["tests_failed"] += 1
        
        self.test_results["test_details"].append({
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def test_1_broker_discovery_panel_component(self):
        """Test 1: Validate broker discovery panel component exists and is properly structured"""
        test_name = "Broker Discovery Panel Component"
        self.logger.info(f"Running {test_name}")
        
        try:
            # Check if the broker discovery panel file exists
            panel_file = "ui_components/broker_discovery_panel.py"
            if not os.path.exists(panel_file):
                self.log_test_result(test_name, False, "Broker discovery panel file not found")
                return
            
            # Try to import and initialize the panel
            config = {
                "refresh_rate": {"broker_discovery": 15},
                "modules_to_monitor": ["BrokerDiscoveryEngine"]
            }
            
            panel = BrokerDiscoveryPanel(config)
            
            # Check essential methods exist
            required_methods = [
                "load_broker_discovery_state",
                "render_broker_discovery_panel",
                "get_broker_metrics_summary"
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(panel, method):
                    missing_methods.append(method)
            
            if missing_methods:
                self.log_test_result(test_name, False, f"Missing methods: {missing_methods}")
            else:
                # Check default state structure
                expected_state_keys = [
                    "rule_profile_active",
                    "account_type_detected",
                    "override_mode",
                    "broker_discovery_status",
                    "rule_customization_active"
                ]
                
                missing_keys = [key for key in expected_state_keys if key not in panel.broker_state]
                
                if missing_keys:
                    self.log_test_result(test_name, False, f"Missing state keys: {missing_keys}")
                else:
                    self.log_test_result(test_name, True, "Panel component properly structured with all required methods and state")
                    
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
    
    def test_2_dashboard_configuration_updates(self):
        """Test 2: Validate dashboard configuration has been updated for Phase 34"""
        test_name = "Dashboard Configuration Updates"
        self.logger.info(f"Running {test_name}")
        
        try:
            # Load dashboard configuration
            with open("dashboard_config.json", "r") as f:
                config = json.load(f)
            
            # Check for broker_discovery tab
            tabs = config.get("layout", {}).get("main_content", {}).get("tabs", [])
            if "broker_discovery" not in tabs:
                self.log_test_result(test_name, False, "broker_discovery tab not found in configuration")
                return
            
            # Check for broker_discovery refresh rate
            refresh_rates = config.get("refresh_rate", {})
            if "broker_discovery" not in refresh_rates:
                self.log_test_result(test_name, False, "broker_discovery refresh rate not configured")
                return
            
            # Check for BrokerDiscoveryEngine in modules to monitor
            modules = config.get("modules_to_monitor", [])
            if "BrokerDiscoveryEngine" not in modules:
                self.log_test_result(test_name, False, "BrokerDiscoveryEngine not in modules_to_monitor")
                return
            
            self.log_test_result(test_name, True, "Dashboard configuration properly updated for Phase 34")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
    
    def test_3_dashboard_main_integration(self):
        """Test 3: Validate main dashboard file has been updated with broker discovery panel"""
        test_name = "Dashboard Main Integration"
        self.logger.info(f"Running {test_name}")
          try:
            # Read dashboard.py file with UTF-8 encoding
            with open("dashboard.py", "r", encoding="utf-8") as f:
                dashboard_content = f.read()
            
            # Check for broker discovery panel import
            if "from ui_components.broker_discovery_panel import BrokerDiscoveryPanel" not in dashboard_content:
                self.log_test_result(test_name, False, "BrokerDiscoveryPanel import not found in dashboard.py")
                return
            
            # Check for panel initialization
            if "self.broker_discovery_panel = BrokerDiscoveryPanel(self.config)" not in dashboard_content:
                self.log_test_result(test_name, False, "BrokerDiscoveryPanel initialization not found")
                return
            
            # Check for render method call
            if "self.broker_discovery_panel.render_broker_discovery_panel()" not in dashboard_content:
                self.log_test_result(test_name, False, "Panel render method call not found")
                return
            
            # Check for proper tab structure (should have 7 tabs now)
            tab_count = dashboard_content.count("with tabs[")
            if tab_count < 7:
                self.log_test_result(test_name, False, f"Expected 7 tabs, found {tab_count}")
                return
            
            self.log_test_result(test_name, True, "Dashboard main file properly integrated with broker discovery panel")
            
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
    
    def test_4_build_status_updates(self):
        """Test 4: Validate build status has been updated to reflect Phase 34 completion"""
        test_name = "Build Status Updates"
        self.logger.info(f"Running {test_name}")
        
        try:
            # Load build status
            with open("build_status.json", "r") as f:
                build_status = json.load(f)
            
            # Check for Phase 34 dashboard integration flags
            expected_flags = [
                "phase_34_broker_discovery_complete",
                "phase_34_telemetry_dashboard_integration",
                "phase_34_dashboard_panel_integration",
                "phase_34_broker_discovery_panel_created",
                "phase_34_dashboard_config_updated"
            ]
            
            missing_flags = []
            for flag in expected_flags:
                if not build_status.get(flag, False):
                    missing_flags.append(flag)
            
            # Check if BrokerDiscoveryEngine is in modules_connected
            modules_connected = build_status.get("modules_connected", [])
            if "BrokerDiscoveryEngine" not in modules_connected:
                missing_flags.append("BrokerDiscoveryEngine not in modules_connected")
            
            if missing_flags:
                self.log_test_result(test_name, False, f"Missing build status flags: {missing_flags}")
            else:
                self.log_test_result(test_name, True, "Build status properly updated for Phase 34 completion")
                
        except Exception as e:
            self.log_test_result(test_name, False, f"Exception: {str(e)}")
    
    def save_test_results(self):
        """Save test results to JSON file"""
        self.test_results["end_time"] = datetime.utcnow().isoformat()
        
        results_filename = f"test_results_phase34_dashboard_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"Detailed Report: {results_filename}")
    
    def run_all_tests(self):
        """Run all Phase 34 dashboard integration tests"""
        try:
            # Run all tests
            self.test_1_broker_discovery_panel_component()
            self.test_2_dashboard_configuration_updates()
            self.test_3_dashboard_main_integration()
            self.test_4_build_status_updates()
            
        except Exception as e:
            self.logger.error(f"Test suite error: {str(e)}")
        
        # Generate final report
        self.logger.info("="*80)
        self.logger.info("PHASE 34 DASHBOARD INTEGRATION TEST RESULTS")
        self.logger.info("="*80)
        self.logger.info(f"Tests Run: {self.test_results['tests_run']}")
        self.logger.info(f"Tests Passed: {self.test_results['tests_passed']}")
        self.logger.info(f"Tests Failed: {self.test_results['tests_failed']}")
        
        success_rate = (self.test_results['tests_passed'] / self.test_results['tests_run'] * 100) if self.test_results['tests_run'] > 0 else 0
        self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        self.save_test_results()
        
        if self.test_results['tests_failed'] > 0:
            self.logger.info(f"WARNING: {self.test_results['tests_failed']} tests failed - Review details above")
            return False
        else:
            self.logger.info("SUCCESS: All Phase 34 dashboard integration tests passed!")
            return True

if __name__ == "__main__":
    test_suite = Phase34DashboardIntegrationTest()
    success = test_suite.run_all_tests()
    
    if success:
        print("SUCCESS: Phase 34 Dashboard Integration validation PASSED")
        sys.exit(0)
    else:
        print("FAILED: Phase 34 Dashboard Integration validation FAILED")
        sys.exit(1)

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
        

# <!-- @GENESIS_MODULE_END: test_phase34_dashboard_integration_clean -->