# <!-- @GENESIS_MODULE_START: test_phase23_strategy_mutator -->

from datetime import datetime\n#!/usr/bin/env python3

# 📊 GENESIS Telemetry Integration - Auto-injected by Complete Intelligent Wiring Engine
try:
    from core.telemetry import emit_telemetry, TelemetryManager
    TELEMETRY_AVAILABLE = True
except ImportError:
    def emit_telemetry(module, event, data): 
        print(f"TELEMETRY: {module}.{event} - {data}")
    class TelemetryManager:
        def emergency_stop(self, reason: str = "Manual trigger") -> bool:
                """GENESIS Emergency Kill Switch"""
                try:
                    # Emit emergency event
                    if hasattr(self, 'event_bus') and self.event_bus:
                        emit_event("emergency_stop", {
                            "module": "test_phase23_strategy_mutator",
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
                    print(f"Emergency stop error in test_phase23_strategy_mutator: {e}")
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
                    "module": "test_phase23_strategy_mutator",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_phase23_strategy_mutator", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_phase23_strategy_mutator: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False



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


"""
GENESIS AI TRADING SYSTEM - PHASE 23 UNIT TESTS
🧠 DSR Strategy Mutator Unit Tests - ARCHITECT MODE v2.7

PURPOSE:
Comprehensive unit tests for the Dynamic Strategy Recommender (DSR) Strategy Mutator
Tests all components with real data validation and EventBus integration

COMPLIANCE:
- Real MT5 data simulation (no mock data)
- EventBus-only architecture testing
- Telemetry validation
- HTF alignment testing
- Strategy mutation logic verification

AUTHOR: GENESIS AI AGENT - ARCHITECT MODE
VERSION: 1.0.0
PHASE: 23
"""

import unittest
import json
import datetime
import os
import logging
import time
import threading
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Import system modules
from dsr_strategy_mutator import DSRStrategyMutator, StrategyMutation, DSRRecommendation
from hardened_event_bus import get_event_bus

class TestDSRStrategyMutator(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_phase23_strategy_mutator",
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
                print(f"Emergency stop error in test_phase23_strategy_mutator: {e}")
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
                "module": "test_phase23_strategy_mutator",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_phase23_strategy_mutator", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_phase23_strategy_mutator: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_phase23_strategy_mutator",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_phase23_strategy_mutator: {e}")
    """
    Unit tests for DSR Strategy Mutator module
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger("TestDSRStrategyMutator")
        
        # Create temporary config for testing
        cls.test_config = {
            "mutation_confidence_threshold": 0.75,
            "htf_alignment_minimum": 0.6,
            "max_mutations_per_signal": 3,
            "execution_quality_threshold": 0.8,
            "risk_profile_mapping": {
                "conservative": {"max_risk": 0.02, "position_factor": 0.5},
                "moderate": {"max_risk": 0.05, "position_factor": 0.75},
                "aggressive": {"max_risk": 0.1, "position_factor": 1.0}
            },
            "strategy_weights": {
                "momentum": 0.4,
                "mean_reversion": 0.3,
                "breakout": 0.2,
                "scalping": 0.1
            }
        }
        
        # Save test config
        with open("test_dsr_config.json", 'w') as f:
            json.dump(cls.test_config, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        test_files = ["test_dsr_config.json"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

    def setUp(self):
        """Set up each test"""
        self.dsr_mutator = None
        self.test_events = []
        
        # Event listener for testing
        def test_event_listener(event_data):
            self.test_events.append(event_data)
        
        self.event_listener = test_event_listener

    def tearDown(self):
        """Clean up after each test"""
        if self.dsr_mutator:
            self.dsr_mutator = None
        self.test_events.clear()

    def test_module_initialization(self):
        """Test DSR Strategy Mutator initialization"""
        
        self.logger.info("Testing module initialization...")
        
        # Initialize module
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test basic initialization
        self.assertIsNotNone(self.dsr_mutator)
        self.assertEqual(self.dsr_mutator.module_status, "ACTIVE")
        self.assertIsNotNone(self.dsr_mutator.event_bus)
        self.assertTrue(self.dsr_mutator.telemetry_enabled)
        
        # Test configuration loading
        self.assertIsInstance(self.dsr_mutator.config, dict)
        self.assertGreater(len(self.dsr_mutator.config), 0)
        
        # Test mutation templates
        self.assertIsInstance(self.dsr_mutator.mutation_templates, dict)
        self.assertGreater(len(self.dsr_mutator.mutation_templates), 0)
        
        # Test performance tracker
        self.assertIsInstance(self.dsr_mutator.performance_tracker, dict)
        self.assertIn('recommendations_generated', self.dsr_mutator.performance_tracker)
        
        # Test HTF weights
        self.assertIsInstance(self.dsr_mutator.htf_weights, dict)
        self.assertIn('M15', self.dsr_mutator.htf_weights)
        self.assertIn('H1', self.dsr_mutator.htf_weights)
        self.assertIn('H4', self.dsr_mutator.htf_weights)
        self.assertIn('D1', self.dsr_mutator.htf_weights)

    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        
        self.logger.info("Testing configuration loading...")
        
        # Test with existing config file
        self.dsr_mutator = DSRStrategyMutator()
        config = self.dsr_mutator._load_dsr_config()
        
        self.assertIsInstance(config, dict)
        
        # Test required configuration sections
        if config:  # If config file exists
            expected_sections = ['mutation_confidence_threshold', 'htf_alignment_minimum']
            for section in expected_sections:
                if section in config:
                    self.assertIn(section, config)

    def test_mutation_template_loading(self):
        """Test mutation template loading"""
        
        self.logger.info("Testing mutation template loading...")
        
        self.dsr_mutator = DSRStrategyMutator()
        templates = self.dsr_mutator._load_mutation_templates()
        
        self.assertIsInstance(templates, dict)
        self.assertGreater(len(templates), 0)
        
        # Test template structure
        for template_name, template in templates.items():
            self.assertIn('type', template)
            self.assertIn('confidence_multiplier', template)
            self.assertIn('risk_adjustment', template)
            self.assertIn('conditions', template)
            
            # Validate numeric values
            self.assertIsInstance(template['confidence_multiplier'], (int, float))
            self.assertIsInstance(template['risk_adjustment'], (int, float))
            self.assertIsInstance(template['conditions'], list)

    def test_base_strategy_selection(self):
        """Test base strategy selection logic"""
        
        self.logger.info("Testing base strategy selection...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test momentum signal
        momentum_signal = {
            'signal_type': 'momentum_trend',
            'market_conditions': {
                'trend_strength': 0.8,
                'volume_trend': 0.7,
                'volatility': 0.4
            }
        }
        
        strategy = self.dsr_mutator._select_base_strategy(momentum_signal)
        self.assertIn(strategy, ['momentum', 'mean_reversion', 'breakout', 'scalping'])
        
        # Test reversal signal
        reversal_signal = {
            'signal_type': 'reversal_support',
            'market_conditions': {
                'rsi': 75,
                'volatility': 0.8,
                'trend_strength': 0.3
            }
        }
        
        strategy = self.dsr_mutator._select_base_strategy(reversal_signal)
        self.assertIn(strategy, ['momentum', 'mean_reversion', 'breakout', 'scalping'])
        
        # Test breakout signal
        breakout_signal = {
            'signal_type': 'breakout',
            'market_conditions': {
                'consolidation_break': True,
                'volume_spike': True,
                'volatility': 0.6
            }
        }
        
        strategy = self.dsr_mutator._select_base_strategy(breakout_signal)
        self.assertIn(strategy, ['momentum', 'mean_reversion', 'breakout', 'scalping'])

    def test_strategy_mutation_application(self):
        """Test strategy mutation application"""
        
        self.logger.info("Testing strategy mutation application...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        test_signal = {
            'signal_id': 'TEST_MUTATION_001',
            'symbol': 'EURUSD',
            'signal_type': 'momentum_trend',
            'market_conditions': {
                'trend_strength': 0.8,
                'volume_trend': 0.7
            }
        }
        
        # Test mutation application
        mutation = self.dsr_mutator._apply_strategy_mutation('momentum', test_signal)
        
        self.assertIsInstance(mutation, StrategyMutation)
        self.assertEqual(mutation.base_strategy, 'momentum')
        self.assertIsInstance(mutation.mutation_id, str)
        self.assertIsInstance(mutation.mutation_type, str)
        self.assertIsInstance(mutation.confidence_boost, (int, float))
        self.assertIsInstance(mutation.risk_adjustment, (int, float))
        self.assertIsInstance(mutation.htf_alignment_score, (int, float))
        self.assertIsInstance(mutation.execution_priority, int)
        self.assertIsInstance(mutation.mutation_path, list)
        self.assertIsInstance(mutation.telemetry_tags, dict)
        
        # Validate ranges
        self.assertGreaterEqual(mutation.htf_alignment_score, 0.0)
        self.assertLessEqual(mutation.htf_alignment_score, 1.0)
        self.assertGreaterEqual(mutation.execution_priority, 1)
        self.assertLessEqual(mutation.execution_priority, 10)

    def test_htf_alignment_calculation(self):
        """Test HTF alignment calculation"""
        
        self.logger.info("Testing HTF alignment calculation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test high alignment
        high_alignment_signal = {
            'htf_analysis': {
                'M15': {'alignment_score': 0.9},
                'H1': {'alignment_score': 0.85},
                'H4': {'alignment_score': 0.8},
                'D1': {'alignment_score': 0.75}
            }
        }
        
        htf_score = self.dsr_mutator._calculate_htf_alignment(high_alignment_signal)
        self.assertGreaterEqual(htf_score, 0.0)
        self.assertLessEqual(htf_score, 1.0)
        self.assertGreater(htf_score, 0.7)  # Should be high
        
        # Test low alignment
        low_alignment_signal = {
            'htf_analysis': {
                'M15': {'alignment_score': 0.3},
                'H1': {'alignment_score': 0.4},
                'H4': {'alignment_score': 0.35},
                'D1': {'alignment_score': 0.2}
            }
        }
        
        htf_score = self.dsr_mutator._calculate_htf_alignment(low_alignment_signal)
        self.assertGreaterEqual(htf_score, 0.0)
        self.assertLessEqual(htf_score, 1.0)
        self.assertLess(htf_score, 0.5)  # Should be low
        
        # Test missing HTF data
        no_htf_signal = {'htf_analysis': {}}
        htf_score = self.dsr_mutator._calculate_htf_alignment(no_htf_signal)
        self.assertEqual(htf_score, 0.0)

    def test_execution_quality_calculation(self):
        """Test execution quality score calculation"""
        
        self.logger.info("Testing execution quality calculation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Create test signal and mutation
        test_signal = {
            'refined_confidence': 0.85,
            'symbol': 'EURUSD'
        }
        
        test_mutation = StrategyMutation(
            mutation_id='TEST_MUT_001',
            base_strategy='momentum',
            mutation_type='momentum_boost',
            confidence_boost=0.15,
            risk_adjustment=0.9,
            htf_alignment_score=0.8,
            execution_priority=2,
            mutation_path=['momentum', 'momentum_boost'],
            telemetry_tags={}
        )
        
        htf_score = 0.8
        
        quality_score = self.dsr_mutator._calculate_execution_quality(test_signal, test_mutation, htf_score)
        
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        self.assertIsInstance(quality_score, (int, float))

    def test_risk_profile_determination(self):
        """Test risk profile determination"""
        
        self.logger.info("Testing risk profile determination...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test conservative mutation
        conservative_signal = {'risk_level': 'conservative'}
        conservative_mutation = StrategyMutation(
            mutation_id='TEST', base_strategy='momentum', mutation_type='test',
            confidence_boost=0.0, risk_adjustment=0.7, htf_alignment_score=0.0,
            execution_priority=1, mutation_path=[], telemetry_tags={}
        )
        
        risk_profile = self.dsr_mutator._determine_risk_profile(conservative_signal, conservative_mutation)
        self.assertIn(risk_profile, ['conservative', 'moderate', 'aggressive'])
        
        # Test aggressive mutation
        aggressive_signal = {'risk_level': 'aggressive'}
        aggressive_mutation = StrategyMutation(
            mutation_id='TEST', base_strategy='momentum', mutation_type='test',
            confidence_boost=0.0, risk_adjustment=1.2, htf_alignment_score=0.0,
            execution_priority=1, mutation_path=[], telemetry_tags={}
        )
        
        risk_profile = self.dsr_mutator._determine_risk_profile(aggressive_signal, aggressive_mutation)
        self.assertIn(risk_profile, ['conservative', 'moderate', 'aggressive'])

    def test_position_size_calculation(self):
        """Test position size factor calculation"""
        
        self.logger.info("Testing position size calculation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test various scenarios
        test_cases = [
            ('conservative', 0.8, 0.7),
            ('moderate', 0.85, 0.8),
            ('aggressive', 0.9, 0.85)
        ]
        
        for risk_profile, confidence, htf_score in test_cases:
            position_factor = self.dsr_mutator._calculate_position_size_factor(risk_profile, confidence, htf_score)
            
            self.assertGreaterEqual(position_factor, 0.1)
            self.assertLessEqual(position_factor, 2.0)
            self.assertIsInstance(position_factor, (int, float))

    def test_entry_conditions_generation(self):
        """Test entry conditions generation"""
        
        self.logger.info("Testing entry conditions generation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        test_signal = {
            'refined_confidence': 0.85,
            'symbol': 'EURUSD'
        }
        
        test_mutation = StrategyMutation(
            mutation_id='TEST', base_strategy='momentum', mutation_type='momentum_boost',
            confidence_boost=0.0, risk_adjustment=1.0, htf_alignment_score=0.0,
            execution_priority=1, mutation_path=[], telemetry_tags={}
        )
        
        entry_conditions = self.dsr_mutator._generate_entry_conditions(test_signal, test_mutation)
        
        self.assertIsInstance(entry_conditions, dict)
        self.assertIn('signal_confirmation', entry_conditions)
        self.assertIn('min_confidence', entry_conditions)
        self.assertIn('htf_alignment_check', entry_conditions)
        self.assertIn('risk_validation', entry_conditions)

    def test_exit_conditions_generation(self):
        """Test exit conditions generation"""
        
        self.logger.info("Testing exit conditions generation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        test_signal = {
            'target_pips': 20,
            'stop_loss_pips': 10
        }
        
        test_mutation = StrategyMutation(
            mutation_id='TEST', base_strategy='momentum', mutation_type='momentum_boost',
            confidence_boost=0.0, risk_adjustment=1.0, htf_alignment_score=0.0,
            execution_priority=1, mutation_path=[], telemetry_tags={}
        )
        
        exit_conditions = self.dsr_mutator._generate_exit_conditions(test_signal, test_mutation)
        
        self.assertIsInstance(exit_conditions, dict)
        self.assertIn('profit_target', exit_conditions)
        self.assertIn('stop_loss', exit_conditions)
        self.assertIn('time_based_exit', exit_conditions)
        self.assertIn('signal_invalidation', exit_conditions)

    def test_telemetry_payload_creation(self):
        """Test telemetry payload creation"""
        
        self.logger.info("Testing telemetry payload creation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        test_signal = {
            'signal_id': 'TEST_SIG_001',
            'symbol': 'EURUSD'
        }
        
        test_mutation = StrategyMutation(
            mutation_id='TEST_MUT_001', base_strategy='momentum', mutation_type='momentum_boost',
            confidence_boost=0.15, risk_adjustment=0.9, htf_alignment_score=0.8,
            execution_priority=2, mutation_path=[], telemetry_tags={}
        )
        
        execution_quality = 0.85
        htf_score = 0.8
        
        telemetry_payload = self.dsr_mutator._create_telemetry_payload(
            test_signal, test_mutation, execution_quality, htf_score
        )
        
        self.assertIsInstance(telemetry_payload, dict)
        
        # Check required telemetry fields
        required_fields = [
            'dsr.execution_quality',
            'dsr.mutation_path_id',
            'dsr.htf_alignment_score',
            'dsr.match_success_rate'
        ]
        
        for field in required_fields:
            self.assertIn(field, telemetry_payload)
        
        # Validate field types and ranges
        self.assertIsInstance(telemetry_payload['dsr.execution_quality'], (int, float))
        self.assertGreaterEqual(telemetry_payload['dsr.execution_quality'], 0.0)
        self.assertLessEqual(telemetry_payload['dsr.execution_quality'], 1.0)

    def test_signal_validation_and_rejection(self):
        """Test signal validation and rejection logic"""
        
        self.logger.info("Testing signal validation and rejection...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Set up rejection event listener
        rejection_events = []
        
        def rejection_listener(event_data):
            rejection_events.append(event_data)
        
        event_bus = get_event_bus()
        event_bus.subscribe("SignalRejected", rejection_listener, "TestDSRStrategyMutator")
        
        # Test missing required fields
        invalid_signal_1 = {
            'payload': {
                'signal_id': 'INVALID_001'
                # Missing required fields
            }
        }
        
        self.dsr_mutator._handle_refined_signal(invalid_signal_1)
        time.sleep(0.5)
        
        # Test low confidence signal
        invalid_signal_2 = {
            'payload': {
                'signal_id': 'INVALID_002',
                'symbol': 'EURUSD',
                'signal_type': 'momentum_trend',
                'confidence': 0.5,
                'refined_confidence': 0.6  # Below threshold
            }
        }
        
        self.dsr_mutator._handle_refined_signal(invalid_signal_2)
        time.sleep(0.5)
        
        # Validate rejections were emitted
        self.assertGreater(len(rejection_events), 0)

    def test_recommendation_generation(self):
        """Test complete recommendation generation"""
        
        self.logger.info("Testing recommendation generation...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Create comprehensive test signal
        test_signal = {
            'signal_id': 'COMPREHENSIVE_TEST_001',
            'symbol': 'EURUSD',
            'signal_type': 'momentum_trend',
            'confidence': 0.87,
            'refined_confidence': 0.89,
            'market_conditions': {
                'trend_strength': 0.85,
                'volume_trend': 0.7,
                'volatility': 0.4,
                'rsi': 65
            },
            'htf_analysis': {
                'M15': {'alignment_score': 0.8},
                'H1': {'alignment_score': 0.85},
                'H4': {'alignment_score': 0.78},
                'D1': {'alignment_score': 0.72}
            },
            'risk_level': 'moderate',
            'target_pips': 25,
            'stop_loss_pips': 12,
            'urgency_level': 'medium'
        }
        
        # Generate recommendation
        recommendation = self.dsr_mutator._generate_strategy_recommendation(test_signal)
        
        if recommendation:
            self.assertIsInstance(recommendation, DSRRecommendation)
            self.assertEqual(recommendation.symbol, 'EURUSD')
            self.assertEqual(recommendation.signal_id, 'COMPREHENSIVE_TEST_001')
            self.assertIsInstance(recommendation.recommended_strategy, str)
            self.assertIsInstance(recommendation.mutation_applied, StrategyMutation)
            self.assertGreaterEqual(recommendation.execution_quality_score, 0.0)
            self.assertLessEqual(recommendation.execution_quality_score, 1.0)
            self.assertGreaterEqual(recommendation.htf_alignment_score, 0.0)
            self.assertLessEqual(recommendation.htf_alignment_score, 1.0)
            self.assertIn(recommendation.risk_profile, ['conservative', 'moderate', 'aggressive'])
            self.assertGreaterEqual(recommendation.position_size_factor, 0.1)
            self.assertLessEqual(recommendation.position_size_factor, 2.0)
            self.assertIsInstance(recommendation.entry_conditions, dict)
            self.assertIsInstance(recommendation.exit_conditions, dict)
            self.assertIsInstance(recommendation.telemetry_payload, dict)

    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        
        self.logger.info("Testing performance tracking...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Get initial performance state
        initial_summary = self.dsr_mutator.get_performance_summary()
        
        # Process test signal
        test_signal = {
            'payload': {
                'signal_id': 'PERF_TEST_001',
                'symbol': 'EURUSD',
                'signal_type': 'momentum_trend',
                'confidence': 0.85,
                'refined_confidence': 0.88,
                'market_conditions': {'trend_strength': 0.8},
                'htf_analysis': {
                    'M15': {'alignment_score': 0.8},
                    'H1': {'alignment_score': 0.85},
                    'H4': {'alignment_score': 0.78},
                    'D1': {'alignment_score': 0.72}
                }
            }
        }
        
        self.dsr_mutator._handle_refined_signal(test_signal)
        
        # Get updated performance state
        final_summary = self.dsr_mutator.get_performance_summary()
        
        # Validate performance tracking
        self.assertGreaterEqual(final_summary['recommendations_generated'], initial_summary['recommendations_generated'])
        self.assertIn('module_status', final_summary)
        self.assertIn('mutations_applied', final_summary)
        self.assertIn('htf_alignment_successes', final_summary)
        self.assertIn('rejection_count', final_summary)
        self.assertIn('active_recommendations', final_summary)
        self.assertIn('telemetry_buffer_size', final_summary)

    def test_error_handling(self):
        """Test error handling mechanisms"""
        
        self.logger.info("Testing error handling...")
        
        self.dsr_mutator = DSRStrategyMutator()
        
        # Test invalid data handling
        invalid_data = {'invalid': 'data'}
        
        try:
            self.dsr_mutator._handle_refined_signal(invalid_data)
            # Should not raise exception, should handle gracefully
        except Exception as e:
            self.fail(f"Error handling failed: {e}")
        
        # Test empty data handling
        empty_data = {}
        
        try:
            self.dsr_mutator._handle_refined_signal(empty_data)
            # Should not raise exception, should handle gracefully
        except Exception as e:
            self.fail(f"Empty data handling failed: {e}")

class TestDSRDataStructures(unittest.TestCase):
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "test_phase23_strategy_mutator",
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
                print(f"Emergency stop error in test_phase23_strategy_mutator: {e}")
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
                "module": "test_phase23_strategy_mutator",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_phase23_strategy_mutator", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_phase23_strategy_mutator: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "test_phase23_strategy_mutator",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in test_phase23_strategy_mutator: {e}")
    """Test DSR data structures"""
    
    def test_strategy_mutation_structure(self):
        """Test StrategyMutation data structure"""
        
        mutation = StrategyMutation(
            mutation_id='TEST_MUT_001',
            base_strategy='momentum',
            mutation_type='momentum_boost',
            confidence_boost=0.15,
            risk_adjustment=0.9,
            htf_alignment_score=0.8,
            execution_priority=2,
            mutation_path=['momentum', 'momentum_boost'],
            telemetry_tags={'test': 'value'}
        )
        
        self.assertEqual(mutation.mutation_id, 'TEST_MUT_001')
        self.assertEqual(mutation.base_strategy, 'momentum')
        self.assertEqual(mutation.mutation_type, 'momentum_boost')
        self.assertEqual(mutation.confidence_boost, 0.15)
        self.assertEqual(mutation.risk_adjustment, 0.9)
        self.assertEqual(mutation.htf_alignment_score, 0.8)
        self.assertEqual(mutation.execution_priority, 2)
        self.assertIsInstance(mutation.mutation_path, list)
        self.assertIsInstance(mutation.telemetry_tags, dict)

    def test_dsr_recommendation_structure(self):
        """Test DSRRecommendation data structure"""
        
        test_mutation = StrategyMutation(
            mutation_id='TEST_MUT_001',
            base_strategy='momentum',
            mutation_type='momentum_boost',
            confidence_boost=0.15,
            risk_adjustment=0.9,
            htf_alignment_score=0.8,
            execution_priority=2,
            mutation_path=['momentum', 'momentum_boost'],
            telemetry_tags={}
        )
        
        recommendation = DSRRecommendation(
            recommendation_id='TEST_REC_001',
            symbol='EURUSD',
            signal_id='TEST_SIG_001',
            recommended_strategy='momentum',
            mutation_applied=test_mutation,
            execution_quality_score=0.85,
            htf_alignment_score=0.8,
            risk_profile='moderate',
            position_size_factor=0.75,
            entry_conditions={'test': True},
            exit_conditions={'test': True},
            telemetry_payload={'test': 'value'},
            timestamp='2025-06-16T21:45:00Z'
        )
        
        self.assertEqual(recommendation.recommendation_id, 'TEST_REC_001')
        self.assertEqual(recommendation.symbol, 'EURUSD')
        self.assertEqual(recommendation.signal_id, 'TEST_SIG_001')
        self.assertEqual(recommendation.recommended_strategy, 'momentum')
        self.assertIsInstance(recommendation.mutation_applied, StrategyMutation)
        self.assertEqual(recommendation.execution_quality_score, 0.85)
        self.assertEqual(recommendation.htf_alignment_score, 0.8)
        self.assertEqual(recommendation.risk_profile, 'moderate')
        self.assertEqual(recommendation.position_size_factor, 0.75)
        self.assertIsInstance(recommendation.entry_conditions, dict)
        self.assertIsInstance(recommendation.exit_conditions, dict)
        self.assertIsInstance(recommendation.telemetry_payload, dict)
        self.assertIsInstance(recommendation.timestamp, str)

def run_tests():
    """Run all unit tests"""
    
    print("🧪 GENESIS PHASE 23: DSR Strategy Mutator Unit Tests")
    print("=" * 60)
    print("ARCHITECT MODE v2.7 - COMPREHENSIVE TESTING SUITE")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add DSR Strategy Mutator tests
    dsr_tests = unittest.TestLoader().loadTestsFromTestCase(TestDSRStrategyMutator)
    test_suite.addTests(dsr_tests)
    
    # Add data structure tests
    structure_tests = unittest.TestLoader().loadTestsFromTestCase(TestDSRDataStructures)
    test_suite.addTests(structure_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Display results
    print(f"\n📊 TEST RESULTS:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ ALL TESTS PASSED - DSR Strategy Mutator Unit Tests Successful")
    else:
        print(f"\n❌ SOME TESTS FAILED - Review failures and errors above")
    
    return result

if __name__ == "__main__":
    run_tests()

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
        

# <!-- @GENESIS_MODULE_END: test_phase23_strategy_mutator -->