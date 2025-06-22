#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 GENESIS INSTITUTIONAL RISK ENGINE - FTMO COMPLIANT DRAWDOWN MANAGEMENT
=========================================================================

@GENESIS_CATEGORY: INSTITUTIONAL.RISK
@GENESIS_TELEMETRY: ENABLED
@GENESIS_EVENTBUS: EMIT+CONSUME

OBJECTIVE: Advanced risk management with FTMO compliance and drawdown-aware scaling
- Real-time FTMO rule enforcement (daily loss, max drawdown, consistency)
- Dynamic position sizing based on drawdown levels
- Multi-layered risk controls (account, trade, portfolio)
- Kill-switch activation on rule violations
- Institutional-grade risk metrics and monitoring
- Advanced correlation-based position limits

FTMO COMPLIANCE RULES:
- Maximum daily loss: 5% of initial balance
- Maximum total drawdown: 10% of initial balance  
- Trailing drawdown: 5% of highest balance reached
- Minimum trading days: 10+ days
- Profit target: 10% of initial balance
- Maximum lot size limits per instrument
- Weekend position holding restrictions

COMPLIANCE: ARCHITECT MODE v3.0 ENFORCED
=========================================================================
"""

import numpy as np
import pandas as pd
import threading
import time
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from decimal import Decimal, ROUND_DOWN

# GENESIS EventBus Integration
try:
    from hardened_event_bus import get_event_bus, emit_event, subscribe_to_event, register_route
except ImportError:
    from event_bus import get_event_bus, emit_event, subscribe_to_event, register_route


# <!-- @GENESIS_MODULE_END: genesis_institutional_risk_engine -->


# <!-- @GENESIS_MODULE_START: genesis_institutional_risk_engine -->

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | GENESIS-RISK | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("genesis_risk_engine")

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class FTMOViolationType(Enum):
    """FTMO violation types"""
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    MAX_DRAWDOWN_EXCEEDED = "max_drawdown_exceeded"
    TRAILING_DRAWDOWN_EXCEEDED = "trailing_drawdown_exceeded"
    CONSISTENCY_RULE_VIOLATED = "consistency_rule_violated"
    WEEKEND_POSITION_HELD = "weekend_position_held"
    LOT_SIZE_EXCEEDED = "lot_size_exceeded"
    PROFIT_TARGET_REACHED = "profit_target_reached"

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    FIXED_LOT = "fixed_lot"
    PERCENT_RISK = "percent_risk"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    DRAWDOWN_SCALED = "drawdown_scaled"

@dataclass
class FTMORules:
    """FTMO challenge rules configuration"""
    initial_balance: float
    max_daily_loss_percent: float = 5.0
    max_total_drawdown_percent: float = 10.0
    trailing_drawdown_percent: float = 5.0
    profit_target_percent: float = 10.0
    min_trading_days: int = 10
    max_lot_size: float = 1.0
    consistency_rule_enabled: bool = True
    weekend_holding_allowed: bool = False
    max_positions_per_symbol: int = 1
    max_total_positions: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AccountMetrics:
    """Real-time account metrics"""
    current_balance: float
    current_equity: float
    initial_balance: float
    highest_balance: float
    daily_pnl: float
    total_pnl: float
    margin_used: float
    margin_level: float
    free_margin: float
    
    # FTMO specific metrics
    max_daily_loss_limit: float
    max_total_drawdown_limit: float
    trailing_drawdown_limit: float
    profit_target: float
    
    # Current levels
    current_daily_loss: float
    current_total_drawdown: float
    current_trailing_drawdown: float
    days_traded: int
    
    # Risk metrics
    risk_level: RiskLevel
    ftmo_compliant: bool
    violations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        return result

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    
    # Risk metrics
    risk_amount: float
    risk_percent: float
    position_value: float
    unrealized_pnl: float
    max_loss_if_stopped: float
    
    # Correlation risk
    correlation_exposure: float
    portfolio_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class RiskDecision:
    """Risk engine decision result"""
    approved: bool
    risk_level: RiskLevel
    position_size_approved: float
    position_size_recommended: float
    violations: List[str]
    warnings: List[str]
    adjustments_made: List[str]
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['risk_level'] = self.risk_level.value
        return result

class GenesisInstitutionalRiskEngine:
    """
    GENESIS Institutional Risk Engine
    
    FTMO-compliant risk management with advanced drawdown controls
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize risk engine with FTMO configuration"""
        self.config = self._load_config(config_path)
        self.ftmo_rules = self._initialize_ftmo_rules()
        self.running = False
        self.lock = threading.Lock()
        
        # Account state
        self.account_metrics = None
        self.positions = {}  # symbol -> PositionRisk
        self.trading_session = {
            'start_date': date.today(),
            'days_traded': 0,
            'daily_pnl_history': {},
            'highest_balance': 0.0,
            'initial_balance': 0.0
        }
        
        # Risk monitoring
        self.risk_violations = []
        self.kill_switch_active = False
        self.emergency_mode = False
        
        # Position sizing
        self.position_sizing_method = PositionSizingMethod.DRAWDOWN_SCALED
        self.correlation_matrix = {}
        
        # Performance tracking
        self.metrics = {
            'positions_evaluated': 0,
            'positions_approved': 0,
            'positions_rejected': 0,
            'risk_violations': 0,
            'kill_switch_activations': 0,
            'emergency_stops': 0,
            'last_risk_check': None,
            'average_position_risk': 0.0,
            'max_drawdown_today': 0.0,
            'ftmo_compliance_score': 100.0
        }
        
        # EventBus registration
        self._register_event_routes()
        
        logger.info("🧠 GENESIS Institutional Risk Engine initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load risk engine configuration"""
        default_config = {
            'initial_balance': 100000.0,
            'max_daily_loss_percent': 5.0,
            'max_total_drawdown_percent': 10.0,
            'trailing_drawdown_percent': 5.0,
            'profit_target_percent': 10.0,
            'default_risk_per_trade': 1.0,
            'max_risk_per_trade': 2.0,
            'max_portfolio_risk': 10.0,
            'correlation_threshold': 0.7,
            'volatility_adjustment': True,
            'position_sizing_method': 'drawdown_scaled',
            'risk_monitoring_interval': 10,  # seconds
            'telemetry_interval': 60,
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
            'max_lot_sizes': {
                'EURUSD': 1.0,
                'GBPUSD': 1.0,
                'USDJPY': 1.0,
                'AUDUSD': 1.0,
                'default': 0.5
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                default_config.update(config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config

    def _initialize_ftmo_rules(self) -> FTMORules:
        """Initialize FTMO rules from configuration"""
        return FTMORules(
            initial_balance=self.config.get('initial_balance', 100000.0),
            max_daily_loss_percent=self.config.get('max_daily_loss_percent', 5.0),
            max_total_drawdown_percent=self.config.get('max_total_drawdown_percent', 10.0),
            trailing_drawdown_percent=self.config.get('trailing_drawdown_percent', 5.0),
            profit_target_percent=self.config.get('profit_target_percent', 10.0),
            max_lot_size=self.config.get('max_lot_sizes', {}).get('default', 1.0),
            consistency_rule_enabled=self.config.get('consistency_rule_enabled', True),
            weekend_holding_allowed=self.config.get('weekend_holding_allowed', False)
        )

    def _register_event_routes(self):
        """Register EventBus routes for institutional compliance"""
        try:
            # Input routes
            register_route("MT5AccountUpdate", "MT5Connector", "RiskEngine")
            register_route("PositionOpened", "ExecutionEngine", "RiskEngine")
            register_route("PositionClosed", "ExecutionEngine", "RiskEngine")
            register_route("TradeRequest", "StrategyEngine", "RiskEngine")
            register_route("MT5SpreadAlert", "MT5Connector", "RiskEngine")
            
            # Output routes
            register_route("RiskDecision", "RiskEngine", "ExecutionEngine")
            register_route("FTMOViolation", "RiskEngine", "TelemetryEngine")
            register_route("KillSwitchActivated", "RiskEngine", "*")
            register_route("EmergencyStop", "RiskEngine", "*")
            register_route("RiskTelemetry", "RiskEngine", "TelemetryEngine")
            
            # Subscribe to events
            subscribe_to_event("MT5AccountUpdate", self._handle_account_update)
            subscribe_to_event("PositionOpened", self._handle_position_opened)
            subscribe_to_event("PositionClosed", self._handle_position_closed)
            subscribe_to_event("TradeRequest", self._handle_trade_request)
            subscribe_to_event("MT5SpreadAlert", self._handle_spread_alert)
            subscribe_to_event("EmergencyShutdown", self._handle_emergency_shutdown)
            
            logger.info("✅ Risk Engine EventBus routes registered")
            
        except Exception as e:
            logger.error(f"❌ Failed to register EventBus routes: {e}")

    def start(self) -> bool:
        """Start risk engine monitoring"""
        try:
            self.running = True
            
            # Start risk monitoring thread
            monitoring_thread = threading.Thread(
                target=self._risk_monitoring_loop,
                name="RiskEngine-Monitoring",
                daemon=True
            )
            monitoring_thread.start()
            
            # Start telemetry thread
            telemetry_thread = threading.Thread(
                target=self._telemetry_loop,
                name="RiskEngine-Telemetry",
                daemon=True
            )
            telemetry_thread.start()
            
            # Start FTMO compliance thread
            ftmo_thread = threading.Thread(
                target=self._ftmo_compliance_loop,
                name="RiskEngine-FTMO",
                daemon=True
            )
            ftmo_thread.start()
            
            logger.info("🚀 GENESIS Risk Engine started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start risk engine: {e}")
            return False

    def _risk_monitoring_loop(self):
        """Main risk monitoring loop"""
        while self.running and not self.kill_switch_active:
            try:
                start_time = time.time()
                
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check FTMO compliance
                self._check_ftmo_compliance()
                
                # Check portfolio risk
                self._check_portfolio_risk()
                
                # Check correlation limits
                self._check_correlation_limits()
                
                # Update position sizing parameters
                self._update_position_sizing()
                
                # Control loop frequency
                processing_time = time.time() - start_time
                sleep_time = max(0, self.config.get('risk_monitoring_interval', 10) - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Error in risk monitoring loop: {e}")
                time.sleep(5)

    def _update_risk_metrics(self):
        """Update real-time risk metrics"""
        try:
            if not self.account_metrics:
                return
            
            with self.lock:
                # Update daily P&L
                today = date.today()
                if today not in self.trading_session['daily_pnl_history']:
                    self.trading_session['daily_pnl_history'][today] = 0.0
                
                # Calculate current drawdowns
                current_balance = self.account_metrics.current_balance
                initial_balance = self.ftmo_rules.initial_balance
                highest_balance = max(self.trading_session['highest_balance'], current_balance)
                
                # Update highest balance
                if current_balance > self.trading_session['highest_balance']:
                    self.trading_session['highest_balance'] = current_balance
                
                # Calculate drawdown metrics
                total_drawdown = ((initial_balance - current_balance) / initial_balance) * 100
                trailing_drawdown = ((highest_balance - current_balance) / highest_balance) * 100 if highest_balance > 0 else 0
                daily_loss = abs(min(0, self.account_metrics.daily_pnl))
                daily_loss_percent = (daily_loss / initial_balance) * 100
                
                # Update account metrics
                self.account_metrics.current_total_drawdown = total_drawdown
                self.account_metrics.current_trailing_drawdown = trailing_drawdown
                self.account_metrics.current_daily_loss = daily_loss_percent
                
                # Update limits
                self.account_metrics.max_daily_loss_limit = self.ftmo_rules.max_daily_loss_percent
                self.account_metrics.max_total_drawdown_limit = self.ftmo_rules.max_total_drawdown_percent
                self.account_metrics.trailing_drawdown_limit = self.ftmo_rules.trailing_drawdown_percent
                self.account_metrics.profit_target = (initial_balance * self.ftmo_rules.profit_target_percent / 100)
                
                # Determine risk level
                self.account_metrics.risk_level = self._calculate_risk_level()
                
                # Update metrics
                self.metrics['last_risk_check'] = datetime.now().isoformat()
                self.metrics['max_drawdown_today'] = max(self.metrics['max_drawdown_today'], daily_loss_percent)
                
        except Exception as e:
            logger.error(f"❌ Error updating risk metrics: {e}")

    def _calculate_risk_level(self) -> RiskLevel:
        """Calculate current risk level"""
        try:
            if not self.account_metrics:
                return RiskLevel.MEDIUM
            
            # Check for critical violations
            if (self.account_metrics.current_daily_loss >= self.ftmo_rules.max_daily_loss_percent * 0.9 or
                self.account_metrics.current_total_drawdown >= self.ftmo_rules.max_total_drawdown_percent * 0.9 or
                self.account_metrics.current_trailing_drawdown >= self.ftmo_rules.trailing_drawdown_percent * 0.9):
                return RiskLevel.CRITICAL
            
            # Check for high risk
            if (self.account_metrics.current_daily_loss >= self.ftmo_rules.max_daily_loss_percent * 0.7 or
                self.account_metrics.current_total_drawdown >= self.ftmo_rules.max_total_drawdown_percent * 0.7 or
                self.account_metrics.current_trailing_drawdown >= self.ftmo_rules.trailing_drawdown_percent * 0.7):
                return RiskLevel.HIGH
            
            # Check for medium risk
            if (self.account_metrics.current_daily_loss >= self.ftmo_rules.max_daily_loss_percent * 0.5 or
                self.account_metrics.current_total_drawdown >= self.ftmo_rules.max_total_drawdown_percent * 0.5 or
                self.account_metrics.current_trailing_drawdown >= self.ftmo_rules.trailing_drawdown_percent * 0.5):
                return RiskLevel.MEDIUM
            
            return RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk level: {e}")
            return RiskLevel.MEDIUM

    def _check_ftmo_compliance(self):
        """Check FTMO rule compliance"""
        try:
            if not self.account_metrics:
                return
            
            violations = []
            
            # Daily loss check
            if self.account_metrics.current_daily_loss >= self.ftmo_rules.max_daily_loss_percent:
                violations.append(FTMOViolationType.DAILY_LOSS_EXCEEDED)
                
            # Total drawdown check
            if self.account_metrics.current_total_drawdown >= self.ftmo_rules.max_total_drawdown_percent:
                violations.append(FTMOViolationType.MAX_DRAWDOWN_EXCEEDED)
                
            # Trailing drawdown check
            if self.account_metrics.current_trailing_drawdown >= self.ftmo_rules.trailing_drawdown_percent:
                violations.append(FTMOViolationType.TRAILING_DRAWDOWN_EXCEEDED)
            
            # Weekend position check
            if not self.ftmo_rules.weekend_holding_allowed:
                current_time = datetime.now()
                if current_time.weekday() >= 5:  # Saturday or Sunday
                    if self.positions:
                        violations.append(FTMOViolationType.WEEKEND_POSITION_HELD)
            
            # Process violations
            if violations:
                self._handle_ftmo_violations(violations)
            
            # Update compliance status
            self.account_metrics.ftmo_compliant = len(violations) == 0
            self.account_metrics.violations = [v.value for v in violations]
            
            # Update compliance score
            compliance_penalties = len(violations) * 10
            self.metrics['ftmo_compliance_score'] = max(0, 100 - compliance_penalties)
            
        except Exception as e:
            logger.error(f"❌ Error checking FTMO compliance: {e}")

    def _handle_ftmo_violations(self, violations: List[FTMOViolationType]):
        """Handle FTMO rule violations"""
        try:
            for violation in violations:
                logger.error(f"🚨 FTMO VIOLATION: {violation.value}")
                
                # Emit violation event
                emit_event("FTMOViolation", {
                    "violation_type": violation.value,
                    "account_metrics": self.account_metrics.to_dict() if self.account_metrics else {},
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical"
                })
                
                # Track violation
                self.risk_violations.append({
                    "type": violation.value,
                    "timestamp": datetime.now().isoformat(),
                    "account_state": self.account_metrics.to_dict() if self.account_metrics else {}
                })
                
                self.metrics['risk_violations'] += 1
                
                # Activate kill switch for serious violations
                if violation in [FTMOViolationType.DAILY_LOSS_EXCEEDED, 
                               FTMOViolationType.MAX_DRAWDOWN_EXCEEDED,
                               FTMOViolationType.TRAILING_DRAWDOWN_EXCEEDED]:
                    self._activate_kill_switch(f"FTMO violation: {violation.value}")
                    
        except Exception as e:
            logger.error(f"❌ Error handling FTMO violations: {e}")

    def _activate_kill_switch(self, reason: str):
        """Activate emergency kill switch"""
        try:
            if self.kill_switch_active:
                return
                
            logger.error(f"🔴 KILL SWITCH ACTIVATED: {reason}")
            
            self.kill_switch_active = True
            self.emergency_mode = True
            self.metrics['kill_switch_activations'] += 1
            
            # Emit kill switch event
            emit_event("KillSwitchActivated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "account_metrics": self.account_metrics.to_dict() if self.account_metrics else {},
                "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()}
            })
            
            # Emit emergency stop
            emit_event("EmergencyStop", {
                "source": "RiskEngine",
                "reason": reason,
                "action": "close_all_positions",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Error activating kill switch: {e}")

    def evaluate_trade_request(self, trade_request: Dict[str, Any]) -> RiskDecision:
        """Evaluate trade request against risk parameters"""
        try:
            symbol = trade_request.get('symbol', '')
            direction = trade_request.get('direction', '')
            requested_size = trade_request.get('size', 0.0)
            entry_price = trade_request.get('entry_price', 0.0)
            stop_loss = trade_request.get('stop_loss', 0.0)
            
            # Initialize decision
            decision = RiskDecision(
                approved=False,
                risk_level=RiskLevel.HIGH,
                position_size_approved=0.0,
                position_size_recommended=0.0,
                violations=[],
                warnings=[],
                adjustments_made=[],
                confidence=0.0
            )
            
            # Check if kill switch is active
            if self.kill_switch_active:
                decision.violations.append("Kill switch is active")
                return decision
            
            # Check account metrics availability
            if not self.account_metrics:
                decision.violations.append("Account metrics not available")
                return decision
            
            # Check FTMO compliance
            if not self.account_metrics.ftmo_compliant:
                decision.violations.append("Account not FTMO compliant")
                return decision
            
            # Check symbol-specific limits
            max_lot_size = self.config.get('max_lot_sizes', {}).get(symbol, self.ftmo_rules.max_lot_size)
            if requested_size > max_lot_size:
                decision.violations.append(f"Requested size {requested_size} exceeds max {max_lot_size} for {symbol}")
                requested_size = max_lot_size
                decision.adjustments_made.append(f"Size reduced to {max_lot_size}")
            
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss) * requested_size
            risk_percent = (risk_amount / self.account_metrics.current_balance) * 100
            
            # Check risk per trade limits
            max_risk_per_trade = self.config.get('max_risk_per_trade', 2.0)
            if risk_percent > max_risk_per_trade:
                # Calculate adjusted size
                max_risk_amount = (self.account_metrics.current_balance * max_risk_per_trade / 100)
                adjusted_size = max_risk_amount / abs(entry_price - stop_loss)
                
                decision.warnings.append(f"Risk {risk_percent:.2f}% exceeds limit {max_risk_per_trade}%")
                decision.adjustments_made.append(f"Size adjusted from {requested_size} to {adjusted_size:.2f}")
                requested_size = adjusted_size
                risk_percent = max_risk_per_trade
            
            # Apply drawdown scaling
            if self.position_sizing_method == PositionSizingMethod.DRAWDOWN_SCALED:
                scaling_factor = self._calculate_drawdown_scaling()
                adjusted_size = requested_size * scaling_factor
                
                if scaling_factor < 1.0:
                    decision.adjustments_made.append(f"Drawdown scaling applied: {scaling_factor:.2f}")
                    requested_size = adjusted_size
            
            # Check portfolio risk
            portfolio_risk = self._calculate_portfolio_risk_with_new_position(
                symbol, direction, requested_size, entry_price, stop_loss
            )
            
            max_portfolio_risk = self.config.get('max_portfolio_risk', 10.0)
            if portfolio_risk > max_portfolio_risk:
                decision.violations.append(f"Portfolio risk {portfolio_risk:.2f}% exceeds limit {max_portfolio_risk}%")
                return decision
            
            # Check correlation limits
            correlation_risk = self._calculate_correlation_risk(symbol, direction, requested_size)
            if correlation_risk > self.config.get('correlation_threshold', 0.7):
                decision.warnings.append(f"High correlation risk: {correlation_risk:.2f}")
            
            # Check position limits
            current_positions = len([p for p in self.positions.values() if p.symbol == symbol])
            if current_positions >= self.ftmo_rules.max_positions_per_symbol:
                decision.violations.append(f"Maximum positions for {symbol} already reached")
                return decision
            
            total_positions = len(self.positions)
            if total_positions >= self.ftmo_rules.max_total_positions:
                decision.violations.append("Maximum total positions reached")
                return decision
            
            # Final approval
            if not decision.violations:
                decision.approved = True
                decision.position_size_approved = requested_size
                decision.position_size_recommended = self._calculate_optimal_position_size(
                    symbol, direction, entry_price, stop_loss
                )
                decision.risk_level = self._calculate_position_risk_level(risk_percent, portfolio_risk)
                decision.confidence = self._calculate_confidence_score(decision)
            
            # Update metrics
            self.metrics['positions_evaluated'] += 1
            if decision.approved:
                self.metrics['positions_approved'] += 1
            else:
                self.metrics['positions_rejected'] += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Error evaluating trade request: {e}")
            return RiskDecision(
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                position_size_approved=0.0,
                position_size_recommended=0.0,
                violations=[f"Error in risk evaluation: {str(e)}"],
                warnings=[],
                adjustments_made=[],
                confidence=0.0
            )

    def _calculate_drawdown_scaling(self) -> float:
        """Calculate position size scaling based on current drawdown"""
        try:
            if not self.account_metrics:
                return 1.0
            
            # Use the worst of daily loss or trailing drawdown
            worst_drawdown = max(
                self.account_metrics.current_daily_loss,
                self.account_metrics.current_trailing_drawdown
            )
            
            # Scaling thresholds
            if worst_drawdown <= 1.0:  # Under 1% drawdown
                return 1.0  # Full size
            elif worst_drawdown <= 2.5:  # 1-2.5% drawdown  
                return 0.8  # 80% size
            elif worst_drawdown <= 4.0:  # 2.5-4% drawdown
                return 0.5  # 50% size
            else:  # Over 4% drawdown
                return 0.2  # 20% size
                
        except Exception as e:
            logger.error(f"❌ Error calculating drawdown scaling: {e}")
            return 0.5

    def _calculate_portfolio_risk_with_new_position(self, symbol: str, direction: str, 
                                                   size: float, entry_price: float, 
                                                   stop_loss: float) -> float:
        """Calculate total portfolio risk including new position"""
        try:
            if not self.account_metrics:
                return 0.0
            
            total_risk = 0.0
            
            # Add existing positions risk
            for pos in self.positions.values():
                total_risk += pos.risk_amount
            
            # Add new position risk
            new_position_risk = abs(entry_price - stop_loss) * size
            total_risk += new_position_risk
            
            # Convert to percentage
            portfolio_risk_percent = (total_risk / self.account_metrics.current_balance) * 100
            
            return portfolio_risk_percent
            
        except Exception as e:
            logger.error(f"❌ Error calculating portfolio risk: {e}")
            return 100.0  # Conservative assumption on error

    def _calculate_correlation_risk(self, symbol: str, direction: str, size: float) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            if not self.positions:
                return 0.0
            
            max_correlation = 0.0
            
            for pos in self.positions.values():
                # Skip same symbol
                if pos.symbol == symbol:
                    continue
                
                # Get correlation coefficient (simplified)
                correlation = self._get_symbol_correlation(symbol, pos.symbol)
                
                # Adjust for direction
                if direction != pos.direction:
                    correlation = -correlation
                
                # Weight by position sizes
                size_weight = (size + pos.size) / 2
                weighted_correlation = abs(correlation) * size_weight
                
                max_correlation = max(max_correlation, weighted_correlation)
            
            return max_correlation
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlation risk: {e}")
            return 1.0

    def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation coefficient between two symbols"""
        # Simplified correlation matrix (in production, this would be calculated from historical data)
        correlations = {
            ('EURUSD', 'GBPUSD'): 0.7,
            ('EURUSD', 'USDCHF'): -0.8,
            ('GBPUSD', 'USDCHF'): -0.6,
            ('USDJPY', 'USDCHF'): 0.5,
            ('AUDUSD', 'NZDUSD'): 0.8,
            ('EURUSD', 'AUDUSD'): 0.4,
            ('GBPUSD', 'AUDUSD'): 0.5
        }
        
        # Try both directions
        key1 = (symbol1, symbol2)
        key2 = (symbol2, symbol1)
        
        if key1 in correlations:
            return correlations[key1]
        elif key2 in correlations:
            return correlations[key2]
        else:
            return 0.0  # No correlation data

    def _calculate_optimal_position_size(self, symbol: str, direction: str, 
                                       entry_price: float, stop_loss: float) -> float:
        """Calculate optimal position size using selected method"""
        try:
            if not self.account_metrics:
                return 0.0
            
            # Base risk per trade (considering current drawdown)
            base_risk_percent = self.config.get('default_risk_per_trade', 1.0)
            scaling_factor = self._calculate_drawdown_scaling()
            adjusted_risk_percent = base_risk_percent * scaling_factor
            
            # Calculate position size
            risk_amount = self.account_metrics.current_balance * adjusted_risk_percent / 100
            pip_value = abs(entry_price - stop_loss)
            
            if pip_value == 0:
                return 0.0
            
            optimal_size = risk_amount / pip_value
            
            # Apply maximum limits
            max_lot_size = self.config.get('max_lot_sizes', {}).get(symbol, self.ftmo_rules.max_lot_size)
            optimal_size = min(optimal_size, max_lot_size)
            
            return round(optimal_size, 2)
            
        except Exception as e:
            logger.error(f"❌ Error calculating optimal position size: {e}")
            return 0.0

    def _calculate_position_risk_level(self, risk_percent: float, portfolio_risk: float) -> RiskLevel:
        """Calculate risk level for position"""
        if risk_percent >= 2.0 or portfolio_risk >= 8.0:
            return RiskLevel.HIGH
        elif risk_percent >= 1.5 or portfolio_risk >= 6.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_confidence_score(self, decision: RiskDecision) -> float:
        """Calculate confidence score for risk decision"""
        confidence = 1.0
        
        # Reduce confidence for each warning
        confidence -= len(decision.warnings) * 0.1
        
        # Reduce confidence for adjustments
        confidence -= len(decision.adjustments_made) * 0.15
        
        # Reduce confidence based on risk level
        if decision.risk_level == RiskLevel.HIGH:
            confidence -= 0.3
        elif decision.risk_level == RiskLevel.MEDIUM:
            confidence -= 0.1
        
        return max(0.0, min(confidence, 1.0))

    def _check_portfolio_risk(self):
        """Check overall portfolio risk"""
        try:
            if not self.account_metrics or not self.positions:
                return
            
            total_risk = sum(pos.risk_amount for pos in self.positions.values())
            portfolio_risk_percent = (total_risk / self.account_metrics.current_balance) * 100
            
            max_portfolio_risk = self.config.get('max_portfolio_risk', 10.0)
            
            if portfolio_risk_percent > max_portfolio_risk:
                logger.warning(f"⚠️ Portfolio risk {portfolio_risk_percent:.2f}% exceeds limit {max_portfolio_risk}%")
                
                # Emit warning
                emit_event("PortfolioRiskWarning", {
                    "current_risk": portfolio_risk_percent,
                    "limit": max_portfolio_risk,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update average position risk
            if self.positions:
                avg_position_risk = total_risk / len(self.positions)
                self.metrics['average_position_risk'] = avg_position_risk
                
        except Exception as e:
            logger.error(f"❌ Error checking portfolio risk: {e}")

    def _check_correlation_limits(self):
        """Check correlation-based position limits"""
        try:
            if len(self.positions) < 2:
                return
            
            # Check for high correlation clusters
            symbols = list(self.positions.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = abs(self._get_symbol_correlation(symbols[i], symbols[j]))
                    
                    if correlation > self.config.get('correlation_threshold', 0.7):
                        logger.warning(f"⚠️ High correlation between {symbols[i]} and {symbols[j]}: {correlation:.2f}")
                        
                        # Emit correlation warning
                        emit_event("CorrelationWarning", {
                            "symbol1": symbols[i],
                            "symbol2": symbols[j],
                            "correlation": correlation,
                            "threshold": self.config.get('correlation_threshold', 0.7),
                            "timestamp": datetime.now().isoformat()
                        })
                        
        except Exception as e:
            logger.error(f"❌ Error checking correlation limits: {e}")

    def _update_position_sizing(self):
        """Update position sizing parameters based on current conditions"""
        try:
            # This would update position sizing parameters based on:
            # - Current drawdown levels
            # - Market volatility
            # - Account performance
            # - Time of day/week
            pass
            
        except Exception as e:
            logger.error(f"❌ Error updating position sizing: {e}")

    def _handle_account_update(self, event_data):
        """Handle MT5 account updates"""
        try:
            data = event_data.get("data", {})
            account_info = data.get("account_info", {})
            
            if account_info:
                # Update account metrics
                if not self.account_metrics:
                    # Initialize on first update
                    self.trading_session['initial_balance'] = account_info.get('balance', 0.0)
                    self.trading_session['highest_balance'] = account_info.get('balance', 0.0)
                    
                    self.account_metrics = AccountMetrics(
                        current_balance=account_info.get('balance', 0.0),
                        current_equity=account_info.get('equity', 0.0),
                        initial_balance=self.trading_session['initial_balance'],
                        highest_balance=self.trading_session['highest_balance'],
                        daily_pnl=account_info.get('profit', 0.0),
                        total_pnl=account_info.get('profit', 0.0),
                        margin_used=account_info.get('margin', 0.0),
                        margin_level=account_info.get('margin_level', 0.0),
                        free_margin=account_info.get('margin_free', 0.0),
                        max_daily_loss_limit=self.ftmo_rules.max_daily_loss_percent,
                        max_total_drawdown_limit=self.ftmo_rules.max_total_drawdown_percent,
                        trailing_drawdown_limit=self.ftmo_rules.trailing_drawdown_percent,
                        profit_target=(self.trading_session['initial_balance'] * self.ftmo_rules.profit_target_percent / 100),
                        current_daily_loss=0.0,
                        current_total_drawdown=0.0,
                        current_trailing_drawdown=0.0,
                        days_traded=0,
                        risk_level=RiskLevel.LOW,
                        ftmo_compliant=True,
                        violations=[]
                    )
                else:
                    # Update existing metrics
                    self.account_metrics.current_balance = account_info.get('balance', self.account_metrics.current_balance)
                    self.account_metrics.current_equity = account_info.get('equity', self.account_metrics.current_equity)
                    self.account_metrics.daily_pnl = account_info.get('profit', self.account_metrics.daily_pnl)
                    self.account_metrics.margin_used = account_info.get('margin', self.account_metrics.margin_used)
                    self.account_metrics.margin_level = account_info.get('margin_level', self.account_metrics.margin_level)
                    self.account_metrics.free_margin = account_info.get('margin_free', self.account_metrics.free_margin)
                
                logger.info(f"📊 Account updated: Balance={self.account_metrics.current_balance:.2f} "
                           f"Equity={self.account_metrics.current_equity:.2f} "
                           f"Margin Level={self.account_metrics.margin_level:.2f}%")
                
        except Exception as e:
            logger.error(f"❌ Error handling account update: {e}")

    def _handle_position_opened(self, event_data):
        """Handle position opened events"""
        try:
            data = event_data.get("data", {})
            symbol = data.get("symbol")
            
            if symbol:
                position_risk = PositionRisk(
                    symbol=data.get("symbol", ""),
                    direction=data.get("direction", ""),
                    size=data.get("size", 0.0),
                    entry_price=data.get("entry_price", 0.0),
                    current_price=data.get("entry_price", 0.0),
                    stop_loss=data.get("stop_loss", 0.0),
                    take_profit=data.get("take_profit", 0.0),
                    risk_amount=abs(data.get("entry_price", 0.0) - data.get("stop_loss", 0.0)) * data.get("size", 0.0),
                    risk_percent=0.0,
                    position_value=data.get("entry_price", 0.0) * data.get("size", 0.0),
                    unrealized_pnl=0.0,
                    max_loss_if_stopped=abs(data.get("entry_price", 0.0) - data.get("stop_loss", 0.0)) * data.get("size", 0.0),
                    correlation_exposure=0.0,
                    portfolio_weight=0.0
                )
                
                # Calculate risk percent
                if self.account_metrics:
                    position_risk.risk_percent = (position_risk.risk_amount / self.account_metrics.current_balance) * 100
                
                with self.lock:
                    self.positions[symbol] = position_risk
                
                logger.info(f"📈 Position opened: {symbol} {position_risk.direction} "
                           f"size={position_risk.size} risk={position_risk.risk_percent:.2f}%")
                
        except Exception as e:
            logger.error(f"❌ Error handling position opened: {e}")

    def _handle_position_closed(self, event_data):
        """Handle position closed events"""
        try:
            data = event_data.get("data", {})
            symbol = data.get("symbol")
            
            if symbol and symbol in self.positions:
                with self.lock:
                    del self.positions[symbol]
                
                logger.info(f"📉 Position closed: {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error handling position closed: {e}")

    def _handle_trade_request(self, event_data):
        """Handle trade request evaluation"""
        try:
            data = event_data.get("data", {})
            
            # Evaluate the trade request
            decision = self.evaluate_trade_request(data)
            
            # Emit risk decision
            emit_event("RiskDecision", {
                "request_id": data.get("request_id", ""),
                "decision": decision.to_dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"⚖️ Risk decision: {data.get('symbol', '')} "
                       f"approved={decision.approved} "
                       f"size={decision.position_size_approved:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error handling trade request: {e}")

    def _handle_spread_alert(self, event_data):
        """Handle spread alerts from MT5 connector"""
        try:
            data = event_data.get("data", {})
            symbol = data.get("symbol")
            spread = data.get("current_spread", 0.0)
            
            # Adjust risk for high spread conditions
            if spread > 5.0:  # High spread threshold
                logger.warning(f"⚠️ High spread alert for {symbol}: {spread}")
                
                # Could implement spread-based position size adjustments here
                
        except Exception as e:
            logger.error(f"❌ Error handling spread alert: {e}")

    def _handle_emergency_shutdown(self, event_data):
        """Handle emergency shutdown commands"""
        logger.warning("🚨 Emergency shutdown received - activating kill switch")
        self._activate_kill_switch("Emergency shutdown command received")

    def _ftmo_compliance_loop(self):
        """FTMO compliance monitoring loop"""
        while self.running and not self.kill_switch_active:
            try:
                # Additional FTMO-specific checks
                self._check_profit_target()
                self._check_trading_days()
                self._check_consistency_rule()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in FTMO compliance loop: {e}")
                time.sleep(60)

    def _check_profit_target(self):
        """Check if profit target has been reached"""
        try:
            if not self.account_metrics:
                return
            
            current_profit = self.account_metrics.current_balance - self.account_metrics.initial_balance
            profit_target = self.account_metrics.initial_balance * self.ftmo_rules.profit_target_percent / 100
            
            if current_profit >= profit_target:
                logger.info(f"🎯 Profit target reached: {current_profit:.2f} >= {profit_target:.2f}")
                
                emit_event("FTMOViolation", {
                    "violation_type": FTMOViolationType.PROFIT_TARGET_REACHED.value,
                    "current_profit": current_profit,
                    "profit_target": profit_target,
                    "timestamp": datetime.now().isoformat(),
                    "severity": "info"
                })
                
        except Exception as e:
            logger.error(f"❌ Error checking profit target: {e}")

    def _check_trading_days(self):
        """Check minimum trading days requirement"""
        try:
            current_date = date.today()
            start_date = self.trading_session['start_date']
            days_elapsed = (current_date - start_date).days
            
            # This is a simplified check - in production, would need to track actual trading days
            if days_elapsed >= self.ftmo_rules.min_trading_days:
                logger.info(f"✅ Minimum trading days requirement met: {days_elapsed} >= {self.ftmo_rules.min_trading_days}")
                
        except Exception as e:
            logger.error(f"❌ Error checking trading days: {e}")

    def _check_consistency_rule(self):
        """Check FTMO consistency rule (largest profitable day <= 5% of total profit)"""
        try:
            if not self.ftmo_rules.consistency_rule_enabled:
                return
              # This would need to be implemented with daily P&L tracking
            # Implementation required for daily P&L consistency check
            
        except Exception as e:
            logger.error(f"❌ Error checking consistency rule: {e}")

    def _telemetry_loop(self):
        """Emit telemetry data"""
        while self.running:
            try:
                time.sleep(self.config.get('telemetry_interval', 60))
                self._emit_telemetry()
            except Exception as e:
                logger.error(f"❌ Error in telemetry loop: {e}")

    def _emit_telemetry(self):
        """Emit comprehensive risk telemetry"""
        try:
            telemetry_data = {
                "risk_engine_status": "running" if self.running else "stopped",
                "kill_switch_active": self.kill_switch_active,
                "emergency_mode": self.emergency_mode,
                "account_metrics": self.account_metrics.to_dict() if self.account_metrics else {},
                "ftmo_rules": self.ftmo_rules.to_dict(),
                "active_positions": len(self.positions),
                "positions": {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
                "performance_metrics": self.metrics,
                "recent_violations": self.risk_violations[-10:],  # Last 10 violations
                "timestamp": datetime.now().isoformat()
            }
            
            emit_event("RiskTelemetry", telemetry_data)
            
        except Exception as e:
            logger.error(f"❌ Error emitting telemetry: {e}")

    def stop(self):
        """Stop risk engine"""
        logger.info("🛑 Stopping GENESIS Risk Engine...")
        self.running = False
        logger.info("✅ GENESIS Risk Engine stopped")

def initialize_risk_engine(config_path: Optional[str] = None) -> GenesisInstitutionalRiskEngine:
    """Initialize and return risk engine instance"""
    engine = GenesisInstitutionalRiskEngine(config_path)
    
    # Store reference for access by other modules
    globals()['_risk_engine_instance'] = engine
    
    logger.info("🏛️ GENESIS Institutional Risk Engine ready")
    return engine

def get_risk_engine() -> Optional[GenesisInstitutionalRiskEngine]:
    """Get current risk engine instance"""
    return globals().get('_risk_engine_instance')

def main():
    """Main execution for testing"""
    logger.info("🧠 GENESIS Institutional Risk Engine - Test Mode")
    
    # Initialize engine
    engine = initialize_risk_engine()
    
    try:
        # Start engine
        if engine.start():
            logger.info("✅ Risk engine started successfully")
            
            # Keep running
            while True:
                time.sleep(60)
                # Print stats every minute
                logger.info(f"📊 Positions evaluated: {engine.metrics['positions_evaluated']}")
        else:
            logger.error("❌ Failed to start risk engine")
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping risk engine...")
    finally:
        engine.stop()

if __name__ == "__main__":
    main()


def monitor_drawdown(max_drawdown_percent: float = 5.0, daily_limit_percent: float = 5.0) -> Dict:
    """
    Monitor account drawdown against FTMO limits
    
    Args:
        max_drawdown_percent: Maximum allowed drawdown percentage
        daily_limit_percent: Maximum allowed daily loss percentage
        
    Returns:
        Dictionary with drawdown status information
    """
    try:
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return {"status": "error", "message": "Failed to get account info"}
        
        # Calculate current drawdown
        balance = account_info.balance
        equity = account_info.equity
        
        current_drawdown = (balance - equity) / balance * 100 if balance > 0 else 0
        
        # Get daily high balance
        from_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        positions = mt5.history_deals_get(from_date, datetime.now())
        
        daily_starting_balance = balance - sum([deal.profit for deal in positions])
        daily_loss_percent = (daily_starting_balance - equity) / daily_starting_balance * 100 if daily_starting_balance > 0 else 0
        
        # Prepare result
        result = {
            "status": "ok",
            "current_drawdown_percent": current_drawdown,
            "max_drawdown_percent": max_drawdown_percent,
            "drawdown_level": current_drawdown / max_drawdown_percent,  # 0.0 to 1.0+
            "daily_loss_percent": daily_loss_percent,
            "daily_limit_percent": daily_limit_percent,
            "daily_loss_level": daily_loss_percent / daily_limit_percent,  # 0.0 to 1.0+
            "warnings": []
        }
        
        # Check drawdown thresholds
        if current_drawdown > max_drawdown_percent * 0.7:
            result["warnings"].append(f"Drawdown at {current_drawdown:.2f}% approaching maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "warning"
            
        if current_drawdown > max_drawdown_percent:
            result["warnings"].append(f"CRITICAL: Drawdown of {current_drawdown:.2f}% exceeds maximum of {max_drawdown_percent:.2f}%")
            result["status"] = "critical"
            
        # Check daily loss thresholds
        if daily_loss_percent > daily_limit_percent * 0.7:
            result["warnings"].append(f"Daily loss at {daily_loss_percent:.2f}% approaching limit of {daily_limit_percent:.2f}%")
            result["status"] = "warning"
            
        if daily_loss_percent > daily_limit_percent:
            result["warnings"].append(f"CRITICAL: Daily loss of {daily_loss_percent:.2f}% exceeds limit of {daily_limit_percent:.2f}%")
            result["status"] = "critical"
        
        # Emit events for warnings
        if result["status"] in ["warning", "critical"]:
            emit_event("risk_threshold_warning", {
                "status": result["status"],
                "warnings": result["warnings"],
                "timestamp": datetime.now().isoformat()
            })
            
        return result
        
    except Exception as e:
        logging.error(f"Error monitoring drawdown: {str(e)}")
        return {"status": "error", "message": str(e)}
