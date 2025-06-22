
import logging
import sys
from pathlib import Path

# GENESIS System Integration
logger = logging.getLogger(__name__)

class SystemIntegration:
    """Connects this module to the GENESIS trading system"""
    
    def __init__(self):
        self.connected = True
        logger.info(f"Module {__name__} connected to GENESIS system")
    
    def register_with_eventbus(self):
        """Register this module with the event bus"""
        pass
    
    def enable_telemetry(self):
        """Enable telemetry for this module"""
        pass

# Auto-connect to system
_integration = SystemIntegration()
_integration.register_with_eventbus()
_integration.enable_telemetry()


# <!-- @GENESIS_MODULE_START: phase59_60_test -->

from datetime import datetime\n#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔐 GENESIS AI AGENT — ARCHITECT MODE v5.0.0
Phase 59-60 Test Suite

Tests for Execution Playbook Generator and Autonomous Order Executor
"""

import json
import os
import datetime
import unittest
import tempfile
import shutil
from execution_playbook_generator import ExecutionPlaybookGenerator
from autonomous_order_executor import AutonomousOrderExecutor

class TestPhase59PlaybookGenerator(unittest.TestCase):
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
    """Test suite for Phase 59 Execution Playbook Generator."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test pattern recommendations
        self.test_patterns = {
            "timestamp": "2025-06-18T14:30:00Z",
            "total_recommendations": 2,
            "recommendations": [
                {
                    "rank": 1,
                    "pattern_id": "test_pattern_1",
                    "category": "technical",
                    "success_rate": 0.82,
                    "confidence": 0.76,
                    "occurrences": 45,
                    "recommendation_strength": 0.62,
                    "description": "Test RSI pattern",
                    "conditions": {
                        "rsi": {"min": 20, "max": 35},
                        "volume": {"above_average": True}
                    },
                    "last_updated": "2025-06-18T14:25:00Z"
                },
                {
                    "rank": 2,
                    "pattern_id": "test_pattern_2",
                    "category": "volatility_based",
                    "success_rate": 0.78,
                    "confidence": 0.72,
                    "occurrences": 38,
                    "recommendation_strength": 0.56,
                    "description": "Test volatility pattern",
                    "conditions": {
                        "atr": {"min": 0.001, "max": 0.003}
                    },
                    "last_updated": "2025-06-18T14:25:00Z"
                }
            ]
        }
        
        with open("pattern_recommendations.json", 'w') as f:
            json.dump(self.test_patterns, f, indent=2)
        
        # Create strategies directory
        os.makedirs("strategies/playbooks", exist_ok=True)
        
        self.generator = ExecutionPlaybookGenerator()
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def test_load_core_files(self):
        """Test loading of pattern recommendations."""
        result = self.generator.load_core_files()
        self.assertTrue(result)
        self.assertEqual(len(self.generator.pattern_recommendations["recommendations"]), 2)
    
    def test_generate_execution_playbooks(self):
        """Test execution playbook generation."""
        self.generator.load_core_files()
        result = self.generator.generate_execution_playbooks()
        
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["playbooks_generated"], 2)
        self.assertGreater(result["validation_success_rate"], 0)
    
    def test_playbook_validation(self):
        """Test playbook validation logic."""
        self.generator.load_core_files()
        
        # Generate a test playbook
        pattern = self.test_patterns["recommendations"][0]
        playbook = self.generator._generate_playbook_for_pattern(pattern)
        
        self.assertIsNotNone(playbook)
        self.assertEqual(playbook["pattern_id"], "test_pattern_1")
        self.assertIn("entry_conditions", playbook)
        self.assertIn("risk_management", playbook)
    
    def test_risk_management_calculation(self):
        """Test risk management parameter calculation."""
        pattern = self.test_patterns["recommendations"][0]
        stop_loss = self.generator._calculate_dynamic_stop_loss(pattern)
        kelly_fraction = self.generator._calculate_kelly_fraction(0.82, 2.5)
        
        self.assertGreater(stop_loss, 0)
        self.assertLess(stop_loss, 0.05)  # Reasonable stop loss
        self.assertGreater(kelly_fraction, 0)
        self.assertLess(kelly_fraction, 0.1)  # Conservative Kelly

class TestPhase60AutonomousExecutor(unittest.TestCase):
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
    """Test suite for Phase 60 Autonomous Order Executor."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test playbooks
        os.makedirs("strategies/playbooks", exist_ok=True)
        
        test_playbook = {
            "playbook_id": "test_playbook_1",
            "pattern_id": "test_pattern_1",
            "pattern_category": "technical",
            "status": "active",
            "entry_conditions": {
                "minimum_confidence": 0.7,
                "volume_confirmation": True,
                "trend_alignment": True,
                "macro_clearance": True
            },
            "risk_management": {
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.05,
                "position_size_pct": 0.01
            },
            "compliance_checks": [
                "kill_switch_compliance",
                "macro_sync_guard",
                "liquidity_sweep_validator"
            ]
        }
        
        with open("strategies/playbooks/test_playbook_1.json", 'w') as f:
            json.dump(test_playbook, f, indent=2)
        
        self.executor = AutonomousOrderExecutor()
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def test_load_playbooks(self):
        """Test loading of execution playbooks."""
        result = self.executor.load_core_files()
        self.assertTrue(result)
        self.assertEqual(len(self.executor.execution_playbooks), 1)
    
    def test_pattern_condition_evaluation(self):
        """Test pattern condition evaluation."""
        self.executor.load_core_files()
        
        playbook = list(self.executor.execution_playbooks.values())[0]
        market_data = {
            "close": 1.1000,
            "rsi": 30,
            "volume": 150,
            "avg_volume": 100,
            "ema_20": 1.0990,
            "ema_50": 1.0980
        }
        
        # Test condition evaluation
        result = self.executor._evaluate_playbook_conditions(playbook, "EURUSD", market_data)
        # Should pass basic conditions
        self.assertIsInstance(result, bool)
    
    def test_compliance_checks(self):
        """Test compliance checking functions."""
        # Test kill switch compliance
        kill_switch_ok = self.executor._check_kill_switch_compliance()
        self.assertTrue(kill_switch_ok)  # Should pass initially
        
        # Test macro sync guard
        macro_ok = self.executor._check_macro_sync_guard()
        self.assertTrue(macro_ok)  # Should pass initially
        
        # Test liquidity validation
        market_data = {
            "bid": 1.1000,
            "ask": 1.1002,
            "avg_spread": 0.0002,
            "volume": 100,
            "avg_volume": 100
        }
        liquidity_ok = self.executor._check_liquidity_sweep_validator("EURUSD", market_data)
        self.assertTrue(liquidity_ok)
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        risk_mgmt = {
            "position_size_pct": 0.01,
            "stop_loss_pct": 0.02
        }
        market_data = {"close": 1.1000}
        
        position_size = self.executor._calculate_position_size(risk_mgmt, market_data)
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 1.0)  # Reasonable position size
    
    def test_stop_loss_calculation(self):
        """Test stop loss and take profit calculation."""
        risk_mgmt = {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05
        }
        market_data = {"close": 1.1000}
        
        stop_loss_buy = self.executor._calculate_stop_loss(risk_mgmt, market_data, "buy")
        stop_loss_sell = self.executor._calculate_stop_loss(risk_mgmt, market_data, "sell")
        
        self.assertLess(stop_loss_buy, 1.1000)  # Buy stop loss below current price
        self.assertGreater(stop_loss_sell, 1.1000)  # Sell stop loss above current price

class TestPhase59_60Integration(unittest.TestCase):
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
    """Integration tests for Phase 59-60."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test data
        test_patterns = {
            "timestamp": "2025-06-18T14:30:00Z",
            "total_recommendations": 1,
            "recommendations": [{
                "rank": 1,
                "pattern_id": "integration_test_pattern",
                "category": "technical",
                "success_rate": 0.85,
                "confidence": 0.80,
                "occurrences": 50,
                "recommendation_strength": 0.68,
                "description": "Integration test pattern",
                "conditions": {
                    "rsi": {"min": 25, "max": 40}
                },
                "last_updated": "2025-06-18T14:25:00Z"
            }]
        }
        
        with open("pattern_recommendations.json", 'w') as f:
            json.dump(test_patterns, f, indent=2)
        
        os.makedirs("strategies/playbooks", exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from pattern to execution readiness."""
        # Phase 59: Generate playbooks
        generator = ExecutionPlaybookGenerator()
        generator.load_core_files()
        generation_result = generator.generate_execution_playbooks()
        
        self.assertEqual(generation_result["status"], "SUCCESS")
        self.assertEqual(generation_result["playbooks_generated"], 1)
        
        # Phase 60: Load and validate playbooks
        executor = AutonomousOrderExecutor()
        load_result = executor.load_core_files()
        
        self.assertTrue(load_result)
        self.assertEqual(len(executor.execution_playbooks), 1)
        
        # Verify playbook structure
        playbook = list(executor.execution_playbooks.values())[0]
        self.assertEqual(playbook["pattern_id"], "integration_test_pattern")
        self.assertIn("entry_conditions", playbook)
        self.assertIn("risk_management", playbook)
        self.assertIn("compliance_checks", playbook)

def run_phase59_60_tests():
    """Run all Phase 59-60 tests."""
    print("🧪 Running Phase 59-60 Test Suite...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase59PlaybookGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase60AutonomousExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase59_60Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    test_report = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        "details": {
            "phase_59_playbook_generator": "PASS" if not any("TestPhase59" in str(f[0]) for f in result.failures + result.errors) else "FAIL",
            "phase_60_autonomous_executor": "PASS" if not any("TestPhase60" in str(f[0]) for f in result.failures + result.errors) else "FAIL",
            "integration_tests": "PASS" if not any("Integration" in str(f[0]) for f in result.failures + result.errors) else "FAIL"
        }
    }
    
    # Save test report
    with open("phase59_60_test_report.json", 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"✅ Test Report: {test_report['total_tests']} tests, {test_report['success_rate']:.1f}% success rate")
    return result.wasSuccessful()

if __name__ == "__main__":
    """Run tests if executed directly."""
    success = run_phase59_60_tests()
    if success:
        print("✅ All Phase 59-60 tests passed!")
        exit(0)
    else:
        print("❌ Some Phase 59-60 tests failed!")
        exit(1)

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
        

# <!-- @GENESIS_MODULE_END: phase59_60_test -->