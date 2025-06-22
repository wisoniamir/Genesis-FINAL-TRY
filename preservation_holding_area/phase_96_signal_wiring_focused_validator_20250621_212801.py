
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

                emit_telemetry("phase_96_signal_wiring_focused_validator", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("phase_96_signal_wiring_focused_validator", "position_calculated", {
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
                            "module": "phase_96_signal_wiring_focused_validator",
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
                    print(f"Emergency stop error in phase_96_signal_wiring_focused_validator: {e}")
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
                    "module": "phase_96_signal_wiring_focused_validator",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("phase_96_signal_wiring_focused_validator", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in phase_96_signal_wiring_focused_validator: {e}")
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
GENESIS Phase 96 Focused Signal Wiring Validator
Focuses on critical signal wiring issues that can be automatically resolved.
Provides actionable reports and targeted fixes for immediate improvements.
"""
import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path


# <!-- @GENESIS_MODULE_END: phase_96_signal_wiring_focused_validator -->


# <!-- @GENESIS_MODULE_START: phase_96_signal_wiring_focused_validator -->

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FocusedSignalWiringValidator:
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

            emit_telemetry("phase_96_signal_wiring_focused_validator", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("phase_96_signal_wiring_focused_validator", "position_calculated", {
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
                        "module": "phase_96_signal_wiring_focused_validator",
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
                print(f"Emergency stop error in phase_96_signal_wiring_focused_validator: {e}")
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
                "module": "phase_96_signal_wiring_focused_validator",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("phase_96_signal_wiring_focused_validator", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in phase_96_signal_wiring_focused_validator: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "phase_96_signal_wiring_focused_validator",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in phase_96_signal_wiring_focused_validator: {e}")
    """
    Focused Phase 96 Signal Wiring Validator
    Targets critical signal wiring issues for immediate resolution.
    """
    
    def __init__(self, workspace_root: str = "."):
        self.workspace_root = Path(workspace_root).absolute()
        self.critical_issues = []
        self.actionable_fixes = []
        
        # Core files
        self.event_bus_file = self.workspace_root / "event_bus.json"
        self.system_tree_file = self.workspace_root / "system_tree.json"
        self.build_status_file = self.workspace_root / "build_status.json"
        
        logger.info(f"Focused Signal Wiring Validator initialized for workspace: {self.workspace_root}")
    
    def validate_critical_signal_wiring(self) -> Dict[str, Any]:
        """
        Main validation entry point for critical signal wiring issues.
        Returns focused validation report.
        """
        logger.info("🔐 Starting GENESIS Phase 96 Focused Signal Wiring Validation...")
        
        report = {
            "phase": 96,
            "validator": "Focused Signal Wiring Validator",
            "timestamp": datetime.now().isoformat(),
            "status": "running",
            "critical_issues": 0,
            "actionable_fixes": 0,
            "issues": [],
            "fixes": [],
            "summary": {}
        }
        
        try:
            # Step 1: Load core data
            event_bus_data = self._load_event_bus()
            system_tree_data = self._load_system_tree()
            
            if not event_bus_data or not system_tree_data:
                return self._generate_error_report("Failed to load core data files")
            
            # Step 2: Focus on critical issues only
            self._validate_active_signal_routes(event_bus_data, system_tree_data)
            self._check_telemetry_handler_compliance(event_bus_data, system_tree_data)
            self._validate_core_module_handlers(system_tree_data)
            
            # Step 3: Generate actionable fixes
            self._generate_actionable_fixes()
            
            # Step 4: Create focused report
            report.update({
                "status": "completed",
                "critical_issues": len(self.critical_issues),
                "actionable_fixes": len(self.actionable_fixes),
                "issues": self.critical_issues,
                "fixes": self.actionable_fixes,
                "summary": self._generate_summary()
            })
            
            # Step 5: Save focused report
            self._save_focused_report(report)
            self._update_build_status(report)
            
            # Step 6: Check if critical issues resolved
            if len(self.critical_issues) == 0:
                logger.info("✅ Phase 96 Focused Signal Wiring Validation PASSED")
                report["phase_96_complete"] = True
                report["signal_wiring_integrity"] = "validated"
            else:
                logger.warning("⛔️ Phase 96 Signal Wiring Validation FAILED - Critical issues found")
            
            return report
            
        except Exception as e:
            logger.error(f"Focused signal wiring validation failed: {str(e)}")
            return self._generate_error_report(str(e))
    
    def _load_event_bus(self) -> Optional[Dict]:
        """Load event_bus.json"""
        try:
            with open(self.event_bus_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load event_bus.json: {str(e)}")
            return None
    
    def _load_system_tree(self) -> Optional[Dict]:
        """Load system_tree.json"""
        try:
            with open(self.system_tree_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load system_tree.json: {str(e)}")
            return None
    
    def _validate_active_signal_routes(self, event_bus_data: Dict, system_tree_data: Dict):
        """Validate only active signal routes with subscribers"""
        logger.info("🔍 Validating active signal routes...")
        
        routes = event_bus_data.get('routes', {})
        modules = system_tree_data.get('modules', {})
        
        # Focus on routes with active subscribers only
        active_routes = {k: v for k, v in routes.items() if v.get('subscribers')}
        
        for route_name, route_config in active_routes.items():
            subscribers = route_config.get('subscribers', [])
            publisher = route_config.get('publisher')
            
            # Critical Issue 1: Active route with publisher not in system tree
            if publisher and publisher not in modules:
                self.critical_issues.append({
                    "type": "missing_active_publisher",
                    "route": route_name,
                    "publisher": publisher,
                    "subscribers_count": len(subscribers),
                    "severity": "HIGH",
                    "impact": "Active subscribers cannot receive signals from missing publisher"
                })
            
            # Critical Issue 2: Active subscribers not in system tree
            missing_subscribers = [sub for sub in subscribers if sub not in modules]
            if missing_subscribers:
                self.critical_issues.append({
                    "type": "missing_active_subscribers",
                    "route": route_name,
                    "missing_subscribers": missing_subscribers,
                    "missing_count": len(missing_subscribers),
                    "severity": "HIGH",
                    "impact": "Signals published but not received by missing subscribers"
                })
        
        logger.info(f"Found {len([i for i in self.critical_issues if 'active' in i['type']])} active route issues")
    
    def _check_telemetry_handler_compliance(self, event_bus_data: Dict, system_tree_data: Dict):
        """Check telemetry handler compliance for core modules"""
        logger.info("🔍 Checking telemetry handler compliance...")
        
        routes = event_bus_data.get('routes', {})
        modules = system_tree_data.get('modules', {})
        
        # Focus on telemetry routes
        telemetry_routes = {k: v for k, v in routes.items() if '_telemetry' in k}
        
        for route_name, route_config in telemetry_routes.items():
            subscribers = route_config.get('subscribers', [])
            
            # Critical Issue: Telemetry route without telemetry_collector
            if 'telemetry_collector' not in subscribers:
                self.critical_issues.append({
                    "type": "missing_telemetry_collector",
                    "route": route_name,
                    "current_subscribers": subscribers,
                    "severity": "MEDIUM",
                    "impact": "Telemetry data not being collected for this route"
                })
        
        logger.info(f"Found {len([i for i in self.critical_issues if 'telemetry' in i['type']])} telemetry compliance issues")
    
    def _validate_core_module_handlers(self, system_tree_data: Dict):
        """Validate that core modules have basic signal handling capability"""
        logger.info("🔍 Validating core module signal handling...")
        
        # Core modules that should have signal handling
        core_modules = [
            'guardian', 'dashboard', 'execution_engine', 'signal_engine', 
            'backtest_engine', 'telemetry_collector'
        ]
        
        modules = system_tree_data.get('modules', {})
        
        for core_module in core_modules:
            if core_module in modules:
                module_info = modules[core_module]
                file_path = module_info.get('file_path', '')
                
                if file_path:
                    abs_file_path = self._resolve_file_path(file_path)
                    if abs_file_path.exists():
                        if not self._has_basic_signal_handling(abs_file_path):
                            self.critical_issues.append({
                                "type": "missing_core_signal_handling",
                                "module": core_module,
                                "file_path": str(abs_file_path),
                                "severity": "MEDIUM",
                                "impact": f"Core module {core_module} lacks basic signal handling"
                            })
        
        logger.info(f"Found {len([i for i in self.critical_issues if 'core_signal' in i['type']])} core module issues")
    
    def _has_basic_signal_handling(self, file_path: Path) -> bool:
        """Check if file has basic signal handling patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for basic signal handling patterns
            signal_patterns = [
                'event_bus', 'emit(', 'subscribe(', 'on_', 'handle_',
                'EventBus', 'signal', 'handler'
            ]
            
            return any(pattern in content for pattern in signal_patterns)
            
        except Exception:
            return False
    
    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve relative file path to absolute path"""
        clean_path = file_path.replace('.\\', '').replace('./', '')
        return self.workspace_root / clean_path
    
    def _generate_actionable_fixes(self):
        """Generate actionable fixes for critical issues"""
        logger.info("🔧 Generating actionable fixes...")
        
        for issue in self.critical_issues:
            if issue['type'] == 'missing_active_publisher':
                self.actionable_fixes.append({
                    "id": f"fix_96_{len(self.actionable_fixes) + 1}",
                    "type": "add_publisher_to_system_tree",
                    "target": issue['publisher'],
                    "route": issue['route'],
                    "action": f"Add missing publisher {issue['publisher']} to system_tree.json",
                    "priority": "HIGH",
                    "estimated_effort": "LOW"
                })
            
            elif issue['type'] == 'missing_active_subscribers':
                for subscriber in issue['missing_subscribers']:
                    self.actionable_fixes.append({
                        "id": f"fix_96_{len(self.actionable_fixes) + 1}",
                        "type": "add_subscriber_to_system_tree",
                        "target": subscriber,
                        "route": issue['route'],
                        "action": f"Add missing subscriber {subscriber} to system_tree.json",
                        "priority": "HIGH",
                        "estimated_effort": "LOW"
                    })
            
            elif issue['type'] == 'missing_telemetry_collector':
                self.actionable_fixes.append({
                    "id": f"fix_96_{len(self.actionable_fixes) + 1}",
                    "type": "add_telemetry_collector",
                    "target": issue['route'],
                    "action": f"Add telemetry_collector as subscriber to {issue['route']}",
                    "priority": "MEDIUM",
                    "estimated_effort": "LOW"
                })
            
            elif issue['type'] == 'missing_core_signal_handling':
                self.actionable_fixes.append({
                    "id": f"fix_96_{len(self.actionable_fixes) + 1}",
                    "type": "add_basic_signal_handling",
                    "target": issue['module'],
                    "file": issue['file_path'],
                    "action": f"Add basic signal handling to core module {issue['module']}",
                    "priority": "MEDIUM",
                    "estimated_effort": "MEDIUM"
                })
        
        logger.info(f"Generated {len(self.actionable_fixes)} actionable fixes")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary"""
        issue_types = {}
        for issue in self.critical_issues:
            i_type = issue['type']
            issue_types[i_type] = issue_types.get(i_type, 0) + 1
        
        return {
            "total_critical_issues": len(self.critical_issues),
            "issue_breakdown": issue_types,
            "fixes_available": len(self.actionable_fixes),
            "high_priority_issues": len([i for i in self.critical_issues if i.get('severity') == 'HIGH']),
            "recommendation": "Focus on HIGH severity issues first" if issue_types else "No critical issues found"
        }
    
    def _save_focused_report(self, report: Dict):
        """Save focused signal wiring validation report"""
        report_file = self.workspace_root / "phase_96_signal_wiring_focused_report.md"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# GENESIS Phase 96 Focused Signal Wiring Validation Report\n\n")
                f.write(f"**Generated:** {report['timestamp']}\n")
                f.write(f"**Status:** {report['status']}\n")
                f.write(f"**Critical Issues:** {report['critical_issues']}\n")
                f.write(f"**Actionable Fixes:** {report['actionable_fixes']}\n\n")
                
                # Summary section
                summary = report['summary']
                f.write("## 📊 Summary\n\n")
                f.write(f"- **Total Critical Issues:** {summary['total_critical_issues']}\n")
                f.write(f"- **High Priority Issues:** {summary['high_priority_issues']}\n")
                f.write(f"- **Fixes Available:** {summary['fixes_available']}\n")
                f.write(f"- **Recommendation:** {summary['recommendation']}\n\n")
                
                # Critical issues
                if report['issues']:
                    f.write("## 🚨 Critical Issues\n\n")
                    for i, issue in enumerate(report['issues'], 1):
                        f.write(f"### {i}. {issue['type'].replace('_', ' ').title()}\n")
                        f.write(f"- **Severity:** {issue['severity']}\n")
                        f.write(f"- **Impact:** {issue['impact']}\n")
                        for key, value in issue.items():
                            if key not in ['type', 'severity', 'impact']:
                                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                        f.write("\n")
                
                # Actionable fixes
                if report['fixes']:
                    f.write("## 🔧 Actionable Fixes\n\n")
                    for fix in report['fixes']:
                        f.write(f"### {fix['id']} - {fix['type'].replace('_', ' ').title()}\n")
                        f.write(f"- **Target:** {fix['target']}\n")
                        f.write(f"- **Action:** {fix['action']}\n")
                        f.write(f"- **Priority:** {fix['priority']}\n")
                        f.write(f"- **Effort:** {fix['estimated_effort']}\n\n")
                
                # Next steps
                f.write("## 🚀 Next Steps\n\n")
                f.write("1. Address HIGH priority issues first\n")
                f.write("2. Apply suggested fixes in order\n")
                f.write("3. Re-run validator to verify fixes\n")
                f.write("4. Update system_tree.json and event_bus.json as needed\n\n")
            
            logger.info(f"✅ Focused signal wiring report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save focused signal wiring report: {str(e)}")
    
    def _update_build_status(self, report: Dict):
        """Update build_status.json with Phase 96 results"""
        try:
            if self.build_status_file.exists():
                with open(self.build_status_file, 'r', encoding='utf-8') as f:
                    build_status = json.load(f)
            else:
                build_status = {}
            
            # Update with Phase 96 results
            build_status.update({
                "phase_96_signal_wiring_focused": {
                    "timestamp": report['timestamp'],
                    "status": report['status'],
                    "critical_issues": report['critical_issues'],
                    "actionable_fixes": report['actionable_fixes'],
                    "validator_version": "96.0-focused"
                },
                "last_update": report['timestamp']
            })
            
            # Add completion flags if validation passed
            if report.get('phase_96_complete'):
                build_status["phase_96_complete"] = True
                build_status["signal_wiring_integrity"] = "validated"
            else:
                build_status["phase_96_complete"] = False
                build_status["signal_wiring_integrity"] = "requires_attention"
            
            with open(self.build_status_file, 'w', encoding='utf-8') as f:
                json.dump(build_status, f, indent=2)
            
            logger.info("✅ Build status updated with Phase 96 focused results")
            
        except Exception as e:
            logger.error(f"Failed to update build status: {str(e)}")
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report for validation failure"""
        return {
            "phase": 96,
            "validator": "Focused Signal Wiring Validator",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_message,
            "phase_96_complete": False,
            "signal_wiring_integrity": "failed"
        }

def main():
    """Main entry point for Phase 96 focused signal wiring validation"""
    try:
        validator = FocusedSignalWiringValidator()
        report = validator.validate_critical_signal_wiring()
        
        print("\n" + "="*70)
        print("GENESIS PHASE 96 FOCUSED SIGNAL WIRING VALIDATION COMPLETE")
        print("="*70)
        print(f"Status: {report['status']}")
        print(f"Critical Issues: {report['critical_issues']}")
        print(f"Actionable Fixes: {report['actionable_fixes']}")
        
        if report.get('phase_96_complete'):
            print("✅ Phase 96 PASSED - No critical signal wiring issues")
        else:
            print("⛔️ Phase 96 FAILED - Critical signal wiring issues require attention")
            print(f"High Priority Issues: {report['summary']['high_priority_issues']}")
        
        print("\n📄 See phase_96_signal_wiring_focused_report.md for detailed analysis")
        return report
        
    except Exception as e:
        logger.error(f"Phase 96 focused signal wiring validation failed: {str(e)}")
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
