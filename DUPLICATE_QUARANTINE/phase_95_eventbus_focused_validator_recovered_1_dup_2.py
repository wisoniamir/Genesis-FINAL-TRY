
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

                emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", "position_calculated", {
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
                            "module": "phase_95_eventbus_focused_validator_recovered_1",
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
                    print(f"Emergency stop error in phase_95_eventbus_focused_validator_recovered_1: {e}")
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
                    "module": "phase_95_eventbus_focused_validator_recovered_1",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in phase_95_eventbus_focused_validator_recovered_1: {e}")
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


#!/usr/bin/env python3
"""
GENESIS EventBus Link Validator - Phase 95 (Focused Version)
Enforces total connectivity and routing compliance across the GENESIS EventBus.
Generates focused, actionable reports for fixing EventBus violations.
"""
import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path


# <!-- @GENESIS_MODULE_END: phase_95_eventbus_focused_validator_recovered_1 -->


# <!-- @GENESIS_MODULE_START: phase_95_eventbus_focused_validator_recovered_1 -->

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedEventBusValidator:
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

            emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", "position_calculated", {
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
                        "module": "phase_95_eventbus_focused_validator_recovered_1",
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
                print(f"Emergency stop error in phase_95_eventbus_focused_validator_recovered_1: {e}")
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
                "module": "phase_95_eventbus_focused_validator_recovered_1",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("phase_95_eventbus_focused_validator_recovered_1", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in phase_95_eventbus_focused_validator_recovered_1: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "phase_95_eventbus_focused_validator_recovered_1",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in phase_95_eventbus_focused_validator_recovered_1: {e}")
    """
    GENESIS EventBus Validator - Phase 95 (Focused)
    Validates only critical EventBus connectivity issues for actionable fixes.
    """
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).absolute()
        self.critical_violations = []
        self.actionable_patches = []
        
        # Core files to validate
        self.event_bus_file = self.workspace_root / "event_bus.json"
        self.system_tree_file = self.workspace_root / "system_tree.json"
        self.build_status_file = self.workspace_root / "build_status.json"
        
        logger.info(f"Focused EventBus Validator initialized for workspace: {self.workspace_root}")
    
    def validate_critical_issues(self) -> Dict[str, Any]:
        """
        Main validation entry point - focuses on critical issues only.
        Returns focused validation report.
        """
        logger.info("🔐 Starting GENESIS EventBus Phase 95 Focused Validation...")
        
        report = {
            "phase": 95,
            "validator": "Focused EventBus Link Validator",
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "critical_violations": 0,
            "actionable_patches": 0,
            "issues": [],
            "patches": [],
            "summary": {}
        }
        
        try:
            # Step 1: Load core data
            event_bus_data = self._load_event_bus()
            system_tree_data = self._load_system_tree()
            
            if not event_bus_data or not system_tree_data:
                return self._generate_error_report("Failed to load core EventBus data")
            
            # Step 2: Focus on critical issues only
            self._validate_critical_routes(event_bus_data, system_tree_data)
            self._check_missing_eventbus_bindings()
            self._validate_duplicate_topics(event_bus_data)
            
            # Step 3: Generate actionable patches
            self._generate_actionable_patches()
            
            # Step 4: Create focused report
            report.update({
                "status": "completed",
                "critical_violations": len(self.critical_violations),
                "actionable_patches": len(self.actionable_patches),
                "issues": self.critical_violations,
                "patches": self.actionable_patches,
                "summary": self._generate_summary()
            })
            
            # Step 5: Save focused report
            self._save_focused_report(report)
            self._update_build_status(report)
            
            # Step 6: Check if critical issues resolved
            if len(self.critical_violations) == 0:
                logger.info("✅ EventBus Phase 95 Focused Validation PASSED")
                report["phase_95_complete"] = True
                report["event_bus_integrity"] = "validated"
            else:
                logger.warning("⛔️ EventBus Phase 95 Validation FAILED - Critical issues found")
                self._trigger_focused_guardian_alert(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Focused EventBus validation failed: {str(e)}")
            return self._generate_error_report(str(e))
    
    def _load_event_bus(self) -> Optional[Dict]:
        """Load and validate event_bus.json structure"""
        try:
            if not self.event_bus_file.exists():
                return None
            
            with open(self.event_bus_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data.get('routes'), dict):
                return None
            
            logger.info(f"✅ Loaded event_bus.json with {len(data['routes'])} routes")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load event_bus.json: {str(e)}")
            return None
    
    def _load_system_tree(self) -> Optional[Dict]:
        """Load and validate system_tree.json structure"""
        try:
            if not self.system_tree_file.exists():
                return None
            
            with open(self.system_tree_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded system_tree.json")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load system_tree.json: {str(e)}")
            return None
    
    def _validate_critical_routes(self, event_bus_data: Dict, system_tree_data: Dict):
        """Validate only critical routes that are actively used"""
        logger.info("🔍 Validating critical routes...")
        
        routes = event_bus_data.get('routes', {})
        modules = system_tree_data.get('modules', {})
        
        # Focus on routes with actual subscribers (active routes)
        active_routes = {k: v for k, v in routes.items() if v.get('subscribers')}
        
        for route_name, route_config in active_routes.items():
            publisher = route_config.get('publisher')
            subscribers = route_config.get('subscribers', [])
            
            # Critical Issue 1: Publisher missing but has active subscribers
            if publisher and publisher not in modules:
                self.critical_violations.append({
                    "type": "missing_publisher",
                    "route": route_name,
                    "publisher": publisher,
                    "subscribers": len(subscribers),
                    "severity": "HIGH",
                    "impact": "Active subscribers cannot receive events"
                })
            
            # Critical Issue 2: Active subscribers missing from system tree
            missing_subscribers = [sub for sub in subscribers if sub not in modules]
            if missing_subscribers:
                self.critical_violations.append({
                    "type": "missing_subscribers",
                    "route": route_name,
                    "missing_count": len(missing_subscribers),
                    "missing_modules": missing_subscribers,
                    "severity": "HIGH",
                    "impact": "Events published but not consumed"
                })
        
        logger.info(f"Found {len(self.critical_violations)} critical route violations")
    
    def _check_missing_eventbus_bindings(self):
        """Check for critical modules missing EventBus bindings"""
        logger.info("🔍 Checking for missing EventBus bindings in core modules...")
        
        # Critical core modules that must have EventBus bindings
        core_modules = [
            'guardian.py', 'dashboard.py', 'execution_engine.py',
            'signal_engine.py', 'backtest_engine.py'
        ]
        
        for module_file in core_modules:
            module_path = self.workspace_root / module_file
            if module_path.exists():
                if not self._has_eventbus_integration(module_path):
                    self.critical_violations.append({
                        "type": "missing_eventbus_integration",
                        "module": module_file,
                        "severity": "MEDIUM",
                        "impact": "Core module not integrated with EventBus"
                    })
        
        logger.info(f"Checked {len(core_modules)} core modules for EventBus integration")
    
    def _validate_duplicate_topics(self, event_bus_data: Dict):
        """Check for duplicate topic keys that cause routing conflicts"""
        logger.info("🔍 Checking for duplicate topics...")
        
        routes = event_bus_data.get('routes', {})
        topic_map = {}
        
        for route_name, route_config in routes.items():
            topic = route_config.get('topic', '')
            if topic:
                if topic in topic_map:
                    self.critical_violations.append({
                        "type": "duplicate_topic",
                        "topic": topic,
                        "routes": [topic_map[topic], route_name],
                        "severity": "HIGH",
                        "impact": "Routing conflicts - events may go to wrong consumers"
                    })
                else:
                    topic_map[topic] = route_name
        
        logger.info("Duplicate topic validation completed")
    
    def _has_eventbus_integration(self, file_path: Path) -> bool:
        """Check if file has EventBus integration"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for EventBus patterns
            eventbus_patterns = [
                'event_bus', 'emit(', 'subscribe(', 'EventBus', 
                'get_event_bus', 'emit_event', 'subscribe_to_event'
            ]
            
            return any(pattern in content for pattern in eventbus_patterns)
            
        except Exception:
            return False
    
    def _generate_actionable_patches(self):
        """Generate actionable patches for critical violations only"""
        logger.info("🔧 Generating actionable patches...")
        
        for violation in self.critical_violations:
            if violation['type'] == 'missing_publisher':
                self.actionable_patches.append({
                    "id": f"patch_{len(self.actionable_patches) + 1}",
                    "type": "add_publisher",
                    "target": violation['route'],
                    "action": f"Register publisher '{violation['publisher']}' in system_tree.json",
                    "priority": "HIGH",
                    "estimated_effort": "LOW"
                })
            
            elif violation['type'] == 'missing_subscribers':
                self.actionable_patches.append({
                    "id": f"patch_{len(self.actionable_patches) + 1}",
                    "type": "add_subscribers",
                    "target": violation['route'],
                    "action": f"Register {violation['missing_count']} missing subscribers in system_tree.json",
                    "priority": "HIGH",
                    "estimated_effort": "MEDIUM"
                })
            
            elif violation['type'] == 'duplicate_topic':
                self.actionable_patches.append({
                    "id": f"patch_{len(self.actionable_patches) + 1}",
                    "type": "rename_topic",
                    "target": violation['topic'],
                    "action": f"Rename duplicate topic to make unique: {violation['routes']}",
                    "priority": "HIGH",
                    "estimated_effort": "LOW"
                })
            
            elif violation['type'] == 'missing_eventbus_integration':
                self.actionable_patches.append({
                    "id": f"patch_{len(self.actionable_patches) + 1}",
                    "type": "integrate_eventbus",
                    "target": violation['module'],
                    "action": f"Add EventBus integration to {violation['module']}",
                    "priority": "MEDIUM",
                    "estimated_effort": "MEDIUM"
                })
        
        logger.info(f"Generated {len(self.actionable_patches)} actionable patches")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        violation_types = {}
        for violation in self.critical_violations:
            v_type = violation['type']
            violation_types[v_type] = violation_types.get(v_type, 0) + 1
        
        return {
            "total_critical_violations": len(self.critical_violations),
            "violation_breakdown": violation_types,
            "patches_available": len(self.actionable_patches),
            "high_priority_issues": len([v for v in self.critical_violations if v.get('severity') == 'HIGH']),
            "recommendation": "Focus on HIGH severity issues first" if violation_types else "No critical issues found"
        }
    
    def _save_focused_report(self, report: Dict):
        """Save focused EventBus validation report"""
        report_file = self.workspace_root / "eventbus_focused_report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# GENESIS EventBus Phase 95 Focused Validation Report\n\n")
                f.write(f"**Generated:** {report['timestamp']}\n")
                f.write(f"**Status:** {report['status']}\n")
                f.write(f"**Critical Violations:** {report['critical_violations']}\n")
                f.write(f"**Actionable Patches:** {report['actionable_patches']}\n\n")
                
                # Summary section
                summary = report['summary']
                f.write("## 📊 Summary\n\n")
                f.write(f"- **Total Critical Violations:** {summary['total_critical_violations']}\n")
                f.write(f"- **High Priority Issues:** {summary['high_priority_issues']}\n")
                f.write(f"- **Patches Available:** {summary['patches_available']}\n")
                f.write(f"- **Recommendation:** {summary['recommendation']}\n\n")
                
                # Critical violations
                if report['issues']:
                    f.write("## 🚨 Critical Violations\n\n")
                    for i, violation in enumerate(report['issues'], 1):
                        f.write(f"### {i}. {violation['type'].replace('_', ' ').title()}\n")
                        f.write(f"- **Severity:** {violation['severity']}\n")
                        f.write(f"- **Impact:** {violation['impact']}\n")
                        for key, value in violation.items():
                            if key not in ['type', 'severity', 'impact']:
                                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                        f.write("\n")
                
                # Actionable patches
                if report['patches']:
                    f.write("## 🔧 Actionable Patches\n\n")
                    for patch in report['patches']:
                        f.write(f"### {patch['id']} - {patch['type'].replace('_', ' ').title()}\n")
                        f.write(f"- **Target:** {patch['target']}\n")
                        f.write(f"- **Action:** {patch['action']}\n")
                        f.write(f"- **Priority:** {patch['priority']}\n")
                        f.write(f"- **Effort:** {patch['estimated_effort']}\n\n")
                
                # Next steps
                f.write("## 🚀 Next Steps\n\n")
                f.write("1. Address HIGH priority violations first\n")
                f.write("2. Apply suggested patches in order\n")
                f.write("3. Re-run validator to verify fixes\n")
                f.write("4. Update system_tree.json as needed\n\n")
            
            logger.info(f"✅ Focused EventBus report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save focused EventBus report: {str(e)}")
    
    def _update_build_status(self, report: Dict):
        """Update build_status.json with Phase 95 results"""
        try:
            if self.build_status_file.exists():
                with open(self.build_status_file, 'r', encoding='utf-8') as f:
                    build_status = json.load(f)
            else:
                build_status = {}
            
            # Update with Phase 95 results
            build_status.update({
                "phase_95_eventbus_validation": {
                    "timestamp": report['timestamp'],
                    "status": report['status'],
                    "critical_violations": report['critical_violations'],
                    "actionable_patches": report['actionable_patches'],
                    "validator_version": "95.0-focused"
                },
                "last_update": report['timestamp']
            })
            
            # Add completion flags if validation passed
            if report.get('phase_95_complete'):
                build_status["phase_95_complete"] = True
                build_status["event_bus_integrity"] = "validated"
            else:
                build_status["phase_95_complete"] = False
                build_status["event_bus_integrity"] = "requires_attention"
            
            with open(self.build_status_file, 'w', encoding='utf-8') as f:
                json.dump(build_status, f, indent=2)
            
            logger.info("✅ Build status updated with Phase 95 focused results")
            
        except Exception as e:
            logger.error(f"Failed to update build status: {str(e)}")
    
    def _trigger_focused_guardian_alert(self, report: Dict):
        """Trigger Guardian EVENTBUS_ALERT for critical violations"""
        logger.warning("⛔️ Triggering Guardian EVENTBUS_ALERT - Critical violations found")
        
        alert_data = {
            "alert_type": "EVENTBUS_ALERT",
            "timestamp": datetime.now().isoformat(),
            "critical_violations": report['critical_violations'],
            "high_priority_count": report['summary']['high_priority_issues'],
            "actionable_patches": report['actionable_patches'],
            "status": "CRITICAL_VIOLATIONS_DETECTED"
        }
        
        # Save alert file for Guardian pickup
        alert_file = self.workspace_root / "guardian_eventbus_alert.json"
        try:
            with open(alert_file, 'w', encoding='utf-8') as f:
                json.dump(alert_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Guardian alert: {str(e)}")
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report for validation failure"""
        return {
            "phase": 95,
            "validator": "Focused EventBus Link Validator",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_message,
            "phase_95_complete": False,
            "event_bus_integrity": "failed"
        }

def main():
    """Main entry point for Phase 95 focused EventBus validation"""
    try:
        validator = FocusedEventBusValidator()
        report = validator.validate_critical_issues()
        
        print("\n" + "="*60)
        print("GENESIS EVENTBUS PHASE 95 FOCUSED VALIDATION COMPLETE")
        print("="*60)
        print(f"Status: {report['status']}")
        print(f"Critical Violations: {report['critical_violations']}")
        print(f"Actionable Patches: {report['actionable_patches']}")
        
        if report.get('phase_95_complete'):
            print("✅ Phase 95 PASSED - No critical EventBus issues")
        else:
            print("⛔️ Phase 95 FAILED - Critical violations require attention")
            print(f"High Priority Issues: {report['summary']['high_priority_issues']}")
        
        print("\n📄 See eventbus_focused_report.md for detailed analysis")
        return report
        
    except Exception as e:
        logger.error(f"Phase 95 focused validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()


def detect_divergence(price_data: list, indicator_data: list, window: int = 10) -> Dict:
    """
    Detect regular and hidden divergences between price and indicator
    
    Args:
        price_data: List of price values (closing prices)
        indicator_data: List of indicator values (e.g., RSI, MACD)
        window: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence information
    """
    result = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
        "strength": 0.0
    }
    
    # Need at least window + 1 periods of data
    if len(price_data) < window + 1 or len(indicator_data) < window + 1:
        return result
        
    # Get the current and historical points
    current_price = price_data[-1]
    previous_price = min(price_data[-window:-1]) if price_data[-1] > price_data[-2] else max(price_data[-window:-1])
    previous_price_idx = price_data[-window:-1].index(previous_price) + len(price_data) - window
    
    current_indicator = indicator_data[-1]
    previous_indicator = indicator_data[previous_price_idx]
    
    # Check for regular divergences
    # Bullish - Lower price lows but higher indicator lows
    if current_price < previous_price and current_indicator > previous_indicator:
        result["regular_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Higher price highs but lower indicator highs
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["regular_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Check for hidden divergences
    # Bullish - Higher price lows but lower indicator lows
    elif current_price > previous_price and current_indicator < previous_indicator:
        result["hidden_bullish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
        
    # Bearish - Lower price highs but higher indicator highs
    elif current_price < previous_price and current_indicator > previous_indicator:
        result["hidden_bearish"] = True
        result["strength"] = abs((current_indicator - previous_indicator) / previous_indicator)
    
    # Emit divergence event if detected
    if any([result["regular_bullish"], result["regular_bearish"], 
            result["hidden_bullish"], result["hidden_bearish"]]):
        emit_event("divergence_detected", {
            "type": next(k for k, v in result.items() if v is True and k != "strength"),
            "strength": result["strength"],
            "symbol": price_data.symbol if hasattr(price_data, "symbol") else "unknown",
            "timestamp": datetime.now().isoformat()
        })
        
    return result


def setup_event_subscriptions(self):
    """Set up EventBus subscriptions for this UI component"""
    event_bus.subscribe("market_data_updated", self.handle_market_data_update)
    event_bus.subscribe("trade_executed", self.handle_trade_update)
    event_bus.subscribe("position_changed", self.handle_position_update)
    event_bus.subscribe("risk_threshold_warning", self.handle_risk_warning)
    event_bus.subscribe("system_status_changed", self.handle_system_status_update)
    
    # Register with telemetry
    telemetry.log_event(TelemetryEvent(
        category="ui", 
        name="event_subscriptions_setup", 
        properties={"component": self.__class__.__name__}
    ))
