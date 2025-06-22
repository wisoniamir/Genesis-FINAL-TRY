
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
                            "module": "signal_execution_router_recovered_2",
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
                    print(f"Emergency stop error in signal_execution_router_recovered_2: {e}")
                    return False
        def emit_module_telemetry(self, event: str, data: dict = None):
                """GENESIS Module Telemetry Hook"""
                telemetry_data = {
                    "timestamp": datetime.now().isoformat(),
                    "module": "signal_execution_router_recovered_2",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("signal_execution_router_recovered_2", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in signal_execution_router_recovered_2: {e}")
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


# <!-- @GENESIS_MODULE_START: signal_execution_router -->

"""
GENESIS Phase 71: Signal Execution Router
🔐 ARCHITECT MODE v5.0.0 - FULLY COMPLIANT
📡 Trade Routing & Execution Validation Layer

Listens to EventBus for trade_recommendation_generated events and routes
validated recommendations to MT5 connector with comprehensive risk checks.
"""

import json
import os
import time
import threading
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExecutionDecision:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_execution_router_recovered_2",
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
                print(f"Emergency stop error in signal_execution_router_recovered_2: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "signal_execution_router_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("signal_execution_router_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in signal_execution_router_recovered_2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_execution_router_recovered_2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_execution_router_recovered_2: {e}")
    """Execution decision data structure"""
    recommendation_id: str
    symbol: str
    decision: str  # 'execute', 'reject', 'defer'
    reason: str
    execution_id: Optional[str]
    mt5_order_data: Optional[Dict]
    timestamp: str
    validation_checks: Dict[str, bool]
    risk_metrics: Dict[str, float]
    

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
        class SignalExecutionRouter:
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            try:
                # Emit emergency event
                if hasattr(self, 'event_bus') and self.event_bus:
                    emit_event("emergency_stop", {
                        "module": "signal_execution_router_recovered_2",
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
                print(f"Emergency stop error in signal_execution_router_recovered_2: {e}")
                return False
    def emit_module_telemetry(self, event: str, data: dict = None):
            """GENESIS Module Telemetry Hook"""
            telemetry_data = {
                "timestamp": datetime.now().isoformat(),
                "module": "signal_execution_router_recovered_2",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("signal_execution_router_recovered_2", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in signal_execution_router_recovered_2: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "signal_execution_router_recovered_2",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in signal_execution_router_recovered_2: {e}")
    """
    🔐 GENESIS Phase 71: Signal Execution Router
    
    Validates and routes trade recommendations through comprehensive checks:
    - Risk limit validation (position size, correlation, drawdown)
    - Macro conflict detection and filtering
    - Manual override state checking
    - Compliance validation and audit trail
    - MT5 order formatting and routing
    - Real-time monitoring and alerting
    """
    
    def __init__(self, config_path: str = "signal_execution_router_config.json"):
        """Initialize the Signal Execution Router"""
        self.config_path = config_path
        self.config = self._load_config()
        self.active = True
        self.execution_history = []
        self.pending_executions = {}
        self.telemetry_data = defaultdict(float)  # Changed from int to float for numeric calculations
        self.event_bus = None
        self.manual_override_state = False
        
        # Architect mode compliance
        self.module_id = "signal_execution_router"
        self.fingerprint = self._generate_fingerprint()
        self.architect_compliant = True
        self.version = "1.0.0"
        self.phase = 71
        
        logger.info(f"🔐 SignalExecutionRouter initialized - Phase {self.phase} - v{self.version}")
        self._register_telemetry_hooks()
        self._initialize_risk_monitoring()
        
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _load_config(self) -> Dict:
        """Load signal execution router configuration"""
        default_config = {
            "risk_limits": {
                "max_daily_loss": 0.05,  # 5% of account
                "max_position_size": 0.02,  # 2% per trade
                "max_correlation_exposure": 0.10,  # 10% correlated positions
                "max_open_positions": 5,
                "max_symbol_exposure": 0.06  # 6% per symbol
            },
            "validation_thresholds": {
                "min_confidence": 7.0,
                "min_risk_reward": 1.5,
                "max_spread_percent": 0.05,  # 5% of entry price
                "min_liquidity_score": 0.7
            },
            "execution_settings": {
                "execution_delay_ms": 100,
                "retry_attempts": 3,
                "timeout_seconds": 30,
                "slippage_tolerance": 0.0005
            },
            "compliance_settings": {
                "require_audit_trail": True,
                "log_all_decisions": True,
                "alert_on_rejection": True,
                "compliance_check_timeout": 5.0
            },
            "macro_filters": {
                "check_news_events": True,
                "check_market_hours": True,
                "check_volatility_regime": True,
                "check_correlation_matrix": True
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults for missing keys
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
            
    def _generate_fingerprint(self) -> str:
        """Generate unique module fingerprint for architect mode"""
        timestamp = datetime.now(timezone.utc).isoformat()
        content_hash = hash(f"signal_execution_router_{timestamp}_{self.version}")
        return f"ser-{abs(content_hash) % 1000000:06d}-{int(time.time()) % 100000}"
        
    def _register_telemetry_hooks(self):
        """Register telemetry hooks for architect mode compliance"""
        self.telemetry_hooks = [
            "recommendations_received_count",
            "executions_approved_count",
            "executions_rejected_count",
            "executions_deferred_count",
            "risk_check_failures",
            "macro_conflict_rejections",
            "manual_override_blocks",
            "compliance_validation_failures",
            "mt5_routing_latency_ms",
            "validation_processing_time_ms",
            "average_execution_confidence",
            "risk_utilization_percent",
            "correlation_exposure_percent",
            "daily_loss_percent"        ]
        
    def _initialize_risk_monitoring(self):
        """Initialize risk monitoring systems"""
        self.risk_state = {
            'daily_pnl': 0.0,
            'open_positions': 0,
            'correlation_exposure': 0.0,
            'position_sizes': {},
            'symbol_exposure': defaultdict(float)  # Changed to defaultdict for proper access
        }
        
    def connect_event_bus(self, event_bus):
        """Connect to EventBus for architect mode compliance"""
        self.event_bus = event_bus
        if self.event_bus:
            # Subscribe to trade recommendations
            self.event_bus.subscribe("TradeRecommendationGenerated", self._handle_trade_recommendation)
            self.event_bus.subscribe("ManualOverrideStateChange", self._handle_manual_override)
            self.event_bus.subscribe("RiskStateUpdate", self._handle_risk_update)
            self.event_bus.subscribe("MT5ConnectionStatus", self._handle_mt5_status)
            
            logger.info("🔗 SignalExecutionRouter connected to EventBus")
            
    def _handle_trade_recommendation(self, recommendation_data: Dict):
        """Handle incoming trade recommendations"""
        try:
            self.telemetry_data['recommendations_received_count'] += 1
            
            recommendation_id = recommendation_data.get('recommendation_id')
            assert recommendation_id:
                logger.error("No recommendation ID in trade recommendation")
                return
                
            logger.info(f"📨 Processing trade recommendation: {recommendation_id}")
            
            # Process the recommendation through validation pipeline
            execution_decision = self._process_recommendation(recommendation_data)
            
            # Store execution decision
            self.execution_history.append(execution_decision)
            
            # Publish execution decision
            if self.event_bus:
                self.event_bus.publish("ExecutionDecisionMade", asdict(execution_decision))
                
            # Route to MT5 if approved
            if execution_decision.decision == 'execute':
                self._route_to_mt5(execution_decision)
                
        except Exception as e:
            logger.error(f"Error handling trade recommendation: {e}")
            
    def _handle_manual_override(self, data: Dict):
        """Handle manual override state changes"""
        try:
            self.manual_override_state = data.get('override_active', False)
            logger.info(f"🔧 Manual override state: {self.manual_override_state}")
        except Exception as e:
            logger.error(f"Error handling manual override: {e}")
            
    def _handle_risk_update(self, data: Dict):
        """Handle risk state updates"""
        try:
            for key, value in data.items():
                if key in self.risk_state:
                    self.risk_state[key] = value
            logger.debug(f"📊 Risk state updated: {data}")
        except Exception as e:
            logger.error(f"Error handling risk update: {e}")
            
    def _handle_mt5_status(self, data: Dict):
        """Handle MT5 connection status updates"""
        try:
            status = data.get('status', 'unknown')
            logger.debug(f"💹 MT5 status: {status}")
        except Exception as e:
            logger.error(f"Error handling MT5 status: {e}")
            
    def _process_recommendation(self, recommendation: Dict) -> ExecutionDecision:
        """Process a trade recommendation through the validation pipeline"""
        start_time = time.time()
        
        try:
            recommendation_id = recommendation['recommendation_id']
            symbol = recommendation['symbol']
            
            # Initialize validation checks
            validation_checks = {
                'risk_limits': False,
                'macro_conflicts': False,
                'manual_override': False,
                'compliance': False,
                'liquidity': False,
                'spread_check': False
            }
            
            risk_metrics = {}
            decision = 'reject'
            reason = 'Unknown validation failure'
            mt5_order_data = None
            execution_id = None
            
            # 1. Risk Limit Check
            risk_check_result = self.risk_limit_check(recommendation)
            validation_checks['risk_limits'] = risk_check_result['passed']
            risk_metrics.update(risk_check_result['metrics'])
            
            if not validation_checks['risk_limits']:
                reason = f"Risk limit violation: {risk_check_result['reason']}"
                self.telemetry_data['risk_check_failures'] += 1
            else:
                # 2. Macro Conflict Filter
                macro_check_result = self.macro_conflict_filter(recommendation)
                validation_checks['macro_conflicts'] = macro_check_result['passed']
                
                if not validation_checks['macro_conflicts']:
                    reason = f"Macro conflict detected: {macro_check_result['reason']}"
                    self.telemetry_data['macro_conflict_rejections'] += 1
                else:
                    # 3. Manual Override State Check
                    override_check_result = self.manual_override_state_check(recommendation)
                    validation_checks['manual_override'] = override_check_result['passed']
                    
                    if not validation_checks['manual_override']:
                        reason = f"Manual override block: {override_check_result['reason']}"
                        self.telemetry_data['manual_override_blocks'] += 1
                    else:
                        # 4. Compliance Validator
                        compliance_check_result = self.compliance_validator(recommendation)
                        validation_checks['compliance'] = compliance_check_result['passed']
                        
                        if not validation_checks['compliance']:
                            reason = f"Compliance violation: {compliance_check_result['reason']}"
                            self.telemetry_data['compliance_validation_failures'] += 1
                        else:
                            # 5. Additional checks (liquidity, spread)
                            liquidity_passed, spread_passed = self._additional_checks(recommendation)
                            validation_checks['liquidity'] = liquidity_passed
                            validation_checks['spread_check'] = spread_passed
                            
                            if not liquidity_passed:
                                reason = "Insufficient liquidity"
                            elif not spread_passed:
                                reason = "Spread too wide"
                            else:
                                # All checks passed - approve execution
                                decision = 'execute'
                                reason = 'All validation checks passed'
                                execution_id = str(uuid.uuid4())
                                mt5_order_data = self._format_mt5_order(recommendation, execution_id)
                                self.telemetry_data['executions_approved_count'] += 1
                                
                                # Update average confidence
                                confidence = recommendation.get('confidence', 0.0)
                                current_avg = self.telemetry_data.get('average_execution_confidence', 0.0)
                                count = self.telemetry_data['executions_approved_count']
                                self.telemetry_data['average_execution_confidence'] = (
                                    (current_avg * (count - 1) + confidence) / count
                                )
                                
            if decision == 'reject':
                self.telemetry_data['executions_rejected_count'] += 1
                
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.telemetry_data['validation_processing_time_ms'] += processing_time
            
            # Create execution decision
            execution_decision = ExecutionDecision(
                recommendation_id=recommendation_id,
                symbol=symbol,
                decision=decision,
                reason=reason,
                execution_id=execution_id,
                mt5_order_data=mt5_order_data,
                timestamp=datetime.now(timezone.utc).isoformat(),
                validation_checks=validation_checks,
                risk_metrics=risk_metrics
            )
            
            logger.info(f"📋 Execution decision for {symbol}: {decision} - {reason}")
            return execution_decision
            
        except Exception as e:
            logger.error(f"Error processing recommendation: {e}")
            return ExecutionDecision(
                recommendation_id=recommendation.get('recommendation_id', 'unknown'),
                symbol=recommendation.get('symbol', 'unknown'),
                decision='reject',
                reason=f'Processing error: {str(e)}',
                execution_id=None,
                mt5_order_data=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
                validation_checks={},
                risk_metrics={}
            )
            
    def risk_limit_check(self, recommendation: Dict) -> Dict:
        """Perform comprehensive risk limit checking"""
        try:
            risk_limits = self.config['risk_limits']
            symbol = recommendation['symbol']
            direction = recommendation['direction']
            entry = recommendation['entry']
            stop_loss = recommendation['stop_loss']
            
            # Calculate position size (simplified - would use actual account balance)
            account_balance = 10000  # This would come from MT5 account info
            risk_amount = abs(entry - stop_loss)
            position_size = (account_balance * risk_limits['max_position_size']) / risk_amount
            
            metrics = {
                'position_size': position_size,
                'risk_amount': risk_amount,
                'risk_percent': risk_amount / account_balance * 100
            }
            
            # Check daily loss limit
            current_daily_loss = abs(self.risk_state['daily_pnl']) if self.risk_state['daily_pnl'] < 0 else 0
            daily_loss_percent = current_daily_loss / account_balance
            
            if daily_loss_percent >= risk_limits['max_daily_loss'] is not None, "Real data required - no fallbacks allowed"

# <!-- @GENESIS_MODULE_END: signal_execution_router -->