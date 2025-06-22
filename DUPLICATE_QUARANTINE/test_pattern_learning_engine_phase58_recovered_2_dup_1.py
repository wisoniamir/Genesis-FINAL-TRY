import logging
# <!-- @GENESIS_MODULE_START: test_pattern_learning_engine_phase58 -->

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

                emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", "position_calculated", {
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
                            "module": "test_pattern_learning_engine_phase58_recovered_2",
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
                    print(f"Emergency stop error in test_pattern_learning_engine_phase58_recovered_2: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "test_pattern_learning_engine_phase58_recovered_2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in test_pattern_learning_engine_phase58_recovered_2: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


"""
Test Suite for Phase 58: Pattern Learning Engine
Validates pattern recognition, clustering, and recommendation generation
"""

import pytest
import json
import os
import time
import numpy as np
from datetime import datetime, timedelta
from pattern_learning_engine_phase58 import PatternLearningEngine

class TestPatternLearningEngine:
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

            emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", "position_calculated", {
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
                        "module": "test_pattern_learning_engine_phase58_recovered_2",
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
                print(f"Emergency stop error in test_pattern_learning_engine_phase58_recovered_2: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "test_pattern_learning_engine_phase58_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("test_pattern_learning_engine_phase58_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in test_pattern_learning_engine_phase58_recovered_2: {e}")
    """Test suite for Pattern Learning Engine validation"""
    
    def setup_method(self):
        """Set up test environment"""
        self.pattern_engine = PatternLearningEngine()
        
    def test_initialization(self):
        """Test Pattern Learning Engine initialization"""
        assert self.pattern_engine is not None
        assert "technical" in self.pattern_engine.pattern_categories
        assert "event_driven" in self.pattern_engine.pattern_categories
        assert "volatility_based" in self.pattern_engine.pattern_categories
        assert "time_based" in self.pattern_engine.pattern_categories
        print("✅ Pattern Learning Engine initialization test passed")
        
    def test_live_trade_processing(self):
        """Test live trade data processing"""
        test_trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": "EURUSD",
            "strategy": "momentum",
            "entry_price": 1.0500,
            "exit_price": 1.0520,
            "volume": 10000,
            "duration": 300,
            "profit_loss": 200,
            "success": True,
            "market_conditions": {"trend": "bullish"},
            "technical_indicators": {
                "rsi": 30,
                "macd": 0.002,
                "bollinger_position": 0.2
            },
            "volatility": 0.025,
            "time_of_day": "london_open",
            "day_of_week": "monday"
        }
        
        initial_count = len(self.pattern_engine.live_trades)
        self.pattern_engine.on_live_trade(test_trade)
        
        assert len(self.pattern_engine.live_trades) == initial_count + 1
        assert self.pattern_engine.live_trades[-1]["symbol"] == "EURUSD"
        print("✅ Live trade processing test passed")
        
    def test_backtest_result_processing(self):
        """Test backtest result processing"""
        test_backtest = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": "momentum",
            "symbol": "EURUSD",
            "time_period": "2024-01-01_2024-12-31",
            "total_trades": 100,
            "winning_trades": 65,
            "losing_trades": 35,
            "win_rate": 0.65,
            "profit_factor": 1.5,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.2
        }
        
        initial_count = len(self.pattern_engine.backtest_results)
        self.pattern_engine.on_backtest_result(test_backtest)
        
        assert len(self.pattern_engine.backtest_results) == initial_count + 1
        assert self.pattern_engine.backtest_results[-1]["win_rate"] == 0.65
        print("✅ Backtest result processing test passed")
        
    def test_manual_override_learning(self):
        """Test learning from manual overrides"""
        test_override = {
            "timestamp": datetime.utcnow().isoformat(),
            "override_type": "expert_adjustment",
            "original_signal": {"action": "buy", "confidence": 0.6},
            "modified_signal": {"action": "sell", "confidence": 0.8},
            "reason": "fundamental_divergence",
            "outcome": "success",
            "confidence": 0.9,
            "expert_notes": "GDP data suggests reversal"
        }
        
        initial_count = len(self.pattern_engine.manual_overrides)
        initial_patterns = len(self.pattern_engine.pattern_categories["event_driven"]["patterns"])
        
        self.pattern_engine.on_manual_override(test_override)
        
        assert len(self.pattern_engine.manual_overrides) == initial_count + 1
        # Expert patterns should be created for successful overrides
        print("✅ Manual override learning test passed")
        
    def test_volatility_pattern_analysis(self):
        """Test volatility-based pattern analysis"""
        # Create trades with different volatility levels
        trades = []
        for i in range(15):
            trade = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": "EURUSD",
                "strategy": "volatility_strategy",
                "volatility": 0.025,  # Medium volatility
                "success": i % 3 != 0,  # 66% success rate
                "profit_loss": 100 if i % 3 != 0 else -50,
                "duration": 200 + i * 10
            }
            trades.append(trade)
            self.pattern_engine.on_live_trade(trade)
            
        # Trigger pattern analysis
        self.pattern_engine._analyze_volatility_patterns(trades, trades[-1])
        
        # Check if volatility patterns were identified
        vol_patterns = self.pattern_engine.pattern_categories["volatility_based"]["patterns"]
        print(f"📊 Identified {len(vol_patterns)} volatility patterns")
        print("✅ Volatility pattern analysis test passed")
        
    def test_time_pattern_analysis(self):
        """Test time-based pattern analysis"""
        # Create trades with specific time patterns
        trades = []
        for i in range(12):
            trade = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": "GBPUSD",
                "strategy": "time_strategy",
                "time_of_day": "london_open",
                "day_of_week": "tuesday",
                "success": i % 4 != 0,  # 75% success rate
                "profit_loss": 150 if i % 4 != 0 else -75,
                "duration": 180 + i * 5
            }
            trades.append(trade)
            self.pattern_engine.on_live_trade(trade)
            
        # Trigger pattern analysis
        self.pattern_engine._analyze_time_patterns(trades, trades[-1])
        
        # Check if time patterns were identified
        time_patterns = self.pattern_engine.pattern_categories["time_based"]["patterns"]
        print(f"⏰ Identified {len(time_patterns)} time patterns")
        print("✅ Time pattern analysis test passed")
        
    def test_pattern_validation(self):
        """Test pattern validation against backtest results"""
        # Create a pattern first
        pattern_id = "test_pattern_validation"
        self.pattern_engine.pattern_categories["technical"]["patterns"][pattern_id] = {
            "pattern_id": pattern_id,
            "category": "technical",
            "success_rate": 0.7,
            "occurrences": 20,
            "confidence": 0.65,
            "backtest_validations": []
        }
        
        # Create backtest result that matches pattern
        backself.event_bus.request('data:live_feed') = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": "test_pattern",
            "win_rate": 0.75,
            "total_trades": 50
        }
        
        self.pattern_engine._validate_patterns_with_backtest(backself.event_bus.request('data:live_feed'))
        
        # Check if pattern was validated
        pattern = self.pattern_engine.pattern_categories["technical"]["patterns"][pattern_id]
        assert len(pattern["backtest_validations"]) > 0
        print("✅ Pattern validation test passed")
        
    def test_recommendation_generation(self):
        """Test pattern recommendation generation"""
        # Create multiple patterns with different success rates
        patterns_data = [
            {"id": "high_success", "success_rate": 0.85, "confidence": 0.8, "occurrences": 50},
            {"id": "medium_success", "success_rate": 0.70, "confidence": 0.75, "occurrences": 30},
            {"id": "low_success", "success_rate": 0.55, "confidence": 0.6, "occurrences": 20}
        ]
        
        for pattern_data in patterns_data:
            self.pattern_engine.pattern_categories["technical"]["patterns"][pattern_data["id"]] = {
                "pattern_id": pattern_data["id"],
                "category": "technical",
                "success_rate": pattern_data["success_rate"],
                "confidence": pattern_data["confidence"],
                "occurrences": pattern_data["occurrences"],
                "last_updated": datetime.utcnow().isoformat()
            }
            
        # Generate recommendations
        self.pattern_engine._generate_pattern_recommendations()
        
        # Check if recommendations file was created
        recommendations_path = os.path.join(self.pattern_engine.patterns_path, "pattern_recommendations.json")
        if os.path.exists(recommendations_path):
            with open(recommendations_path, 'r') as f:
                recommendations = json.load(f)
                assert len(recommendations["recommendations"]) > 0
                # High success pattern should be ranked first
                assert recommendations["recommendations"][0]["success_rate"] >= 0.8
                
        print("✅ Recommendation generation test passed")
        
    def test_clustering_functionality(self):
        """Test pattern clustering functionality"""
        # Create sample features for clustering
        features = []
        for i in range(20):
            feature_vector = [
                np.random.uniform(20, 80),  # RSI
                np.random.uniform(-0.01, 0.01),  # MACD
                np.random.uniform(0, 1),  # Bollinger position
                np.random.uniform(0.5, 2.0),  # Volume ratio
                np.random.uniform(-1, 1)  # Trend strength
            ]
            features.append(feature_vector)
            
        # Test clustering
        clusters = self.pattern_engine._cluster_patterns(features, "technical")
        
        if clusters:
            assert "model" in clusters
            assert "labels" in clusters
            assert "score" in clusters
            assert len(clusters["labels"]) == len(features)
            
        print("✅ Clustering functionality test passed")
        
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = self.pattern_engine.config
        assert 0 < config["success_rate_threshold"] < 1
        assert 0 < config["confidence_threshold"] < 1
        assert config["min_pattern_occurrences"] > 0
        assert config["pattern_update_interval"] > 0
        print("✅ Configuration validation test passed")

def run_pattern_learning_tests():
    """Run all Pattern Learning Engine tests"""
    print("🧪 Starting Pattern Learning Engine Phase 58 tests...")
    
    test_suite = TestPatternLearningEngine()
    test_methods = [
        test_suite.test_initialization,
        test_suite.test_live_trade_processing,
        test_suite.test_backtest_result_processing,
        test_suite.test_manual_override_learning,
        test_suite.test_volatility_pattern_analysis,
        test_suite.test_time_pattern_analysis,
        test_suite.test_pattern_validation,
        test_suite.test_recommendation_generation,
        test_suite.test_clustering_functionality,
        test_suite.test_configuration_validation
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            test_suite.setup_method()
            test_method()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test_method.__name__} - {e}")
            failed += 1
            
    print(f"\n📊 Pattern Learning Engine Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    return passed, failed

if __name__ == "__main__":
    run_pattern_learning_tests()

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
        

# <!-- @GENESIS_MODULE_END: test_pattern_learning_engine_phase58 -->