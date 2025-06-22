import logging

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

                emit_telemetry("architect_mode_v3_compliance_engine", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("architect_mode_v3_compliance_engine", "position_calculated", {
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
                            "module": "architect_mode_v3_compliance_engine",
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
                    print(f"Emergency stop error in architect_mode_v3_compliance_engine: {e}")
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
                    "module": "architect_mode_v3_compliance_engine",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("architect_mode_v3_compliance_engine", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in architect_mode_v3_compliance_engine: {e}")
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
🔐 GENESIS AI AGENT — ARCHITECT LOCK-IN v3.0 COMPLIANCE ENGINE
Enforces strict compliance with GENESIS architecture rules and guidelines.
NO BYPASSING ALLOWED.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Set


# <!-- @GENESIS_MODULE_END: architect_mode_v3_compliance_engine -->


# <!-- @GENESIS_MODULE_START: architect_mode_v3_compliance_engine -->


class ArchitectModeViolation(Exception):
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

            emit_telemetry("architect_mode_v3_compliance_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("architect_mode_v3_compliance_engine", "position_calculated", {
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
                        "module": "architect_mode_v3_compliance_engine",
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
                print(f"Emergency stop error in architect_mode_v3_compliance_engine: {e}")
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
                "module": "architect_mode_v3_compliance_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("architect_mode_v3_compliance_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in architect_mode_v3_compliance_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "architect_mode_v3_compliance_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in architect_mode_v3_compliance_engine: {e}")
    """Raised when GENESIS architecture rules are violated."""
    pass


class ArchitectV3ComplianceEngine:
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

            emit_telemetry("architect_mode_v3_compliance_engine", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("architect_mode_v3_compliance_engine", "position_calculated", {
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
                        "module": "architect_mode_v3_compliance_engine",
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
                print(f"Emergency stop error in architect_mode_v3_compliance_engine: {e}")
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
                "module": "architect_mode_v3_compliance_engine",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("architect_mode_v3_compliance_engine", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in architect_mode_v3_compliance_engine: {e}")
    def initialize_eventbus(self):
            """GENESIS EventBus Initialization"""
            try:
                self.event_bus = get_event_bus()
                if self.event_bus:
                    emit_event("module_initialized", {
                        "module": "architect_mode_v3_compliance_engine",
                        "timestamp": datetime.now().isoformat(),
                        "status": "active"
                    })
            except Exception as e:
                print(f"EventBus initialization error in architect_mode_v3_compliance_engine: {e}")
    """
    🚨 STRICT ARCHITECT MODE v3.0 COMPLIANCE ENGINE
    Enforces all GENESIS architecture rules without exception.
    """
    
    def __init__(self, workspace_path: str = ""):
        self.workspace_path = workspace_path or os.getcwd()
        self.core_files = [
            "build_status.json",
            "build_tracker.md", 
            "system_tree.json",
            "module_registry.json",
            "event_bus.json",
            "telemetry.json",
            "compliance.json",
            "live_data.json",
            "real_data.json",
            "genesis_config.json",
            "genesis_docs.json",
            "genesis_telemetry.json",
            "genesis_event_bus.json"
        ]
        self.violations = []
        self.repair_needed = []
        self.deletion_candidates = []
        
    def enforce_architect_mode(self) -> Dict[str, Any]:
        """
        🔐 MAIN ENFORCEMENT ENGINE
        Runs complete GENESIS v3.0 architecture compliance check.
        """
        print("🔐 GENESIS ARCHITECT MODE v3.0 - ENFORCEMENT ACTIVE")
        print("🚨 STRICT COMPLIANCE CHECK INITIATED")
        
        try:
            # Load and validate core files
            self.load_and_validate_core_files()
            
            # Verify module connectivity
            if not self.validate_module_connectivity():
                raise ArchitectModeViolation("Disconnected modules found.")
            
            # Validate EventBus routes
            if not self.validate_event_bus_routes():
                raise ArchitectModeViolation("Isolated calls detected in EventBus.")
            
            # Scan for duplicates
            if self.has_duplicates():
                self.mark_as_deletion_candidate()
            
            # Verify real data usage
            if not self.is_using_real_data():
                raise ArchitectModeViolation("Mock data detected in live modules.")
            
            # Check telemetry is active
            if not self.is_telemetry_active():
                raise ArchitectModeViolation("Telemetry not reporting for all modules.")
            
            # Update build status
            self.update_build_status()
            
            # Log actions
            self.log_build_tracker("Architect mode compliance check completed.")
            
            return self.generate_compliance_report()
            
        except Exception as e:
            self.handle_violation(str(e))
            raise ArchitectModeViolation(f"ARCHITECT_LOCK_BROKEN: {str(e)}")
    
    def load_and_validate_core_files(self) -> None:
        """Load and validate all mandatory GENESIS core files."""
        print("📁 Loading and validating core files...")
        
        missing_files = []
        for file_name in self.core_files:
            file_path = os.path.join(self.workspace_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            self.violations.append(f"Missing core files: {missing_files}")
            self.repair_needed.extend(missing_files)
    
    def validate_module_connectivity(self) -> bool:
        """Verify no orphaned modules exist in system_tree.json."""
        print("🔗 Validating module connectivity...")
        
        try:
            system_tree_path = os.path.join(self.workspace_path, "system_tree.json")
            with open(system_tree_path, 'r') as f:
                system_tree = json.load(f)
            
            orphan_count = system_tree.get("genesis_system_metadata", {}).get("orphan_modules", 0)
            connected_count = system_tree.get("genesis_system_metadata", {}).get("categorized_modules", 0)
            
            if orphan_count > 0:
                print(f"⚠️ WARNING: {orphan_count} orphan modules detected")
                self.violations.append(f"Orphan modules detected: {orphan_count}")
                return False
            
            print(f"✅ Module connectivity verified: {connected_count} connected modules")
            return True
            
        except Exception as e:
            self.violations.append(f"Module connectivity validation failed: {str(e)}")
            return False
    
    def validate_event_bus_routes(self) -> bool:
        """Validate all modules use EventBus, no isolated calls."""
        print("🚌 Validating EventBus routes...")
        
        try:
            event_bus_path = os.path.join(self.workspace_path, "event_bus.json")
            with open(event_bus_path, 'r') as f:
                event_bus = json.load(f)
            
            active_routes = event_bus.get("active_routes", {})
            route_count = len(active_routes)
            
            if route_count == 0:
                self.violations.append("No EventBus routes detected")
                return False
            
            print(f"✅ EventBus validation passed: {route_count} active routes")
            return True
            
        except Exception as e:
            self.violations.append(f"EventBus validation failed: {str(e)}")
            return False
    
    def has_duplicates(self) -> bool:
        """Scan for duplicate modules, configurations, or routes."""
        print("🔍 Scanning for duplicates...")
        
        try:
            module_registry_path = os.path.join(self.workspace_path, "module_registry.json")
            with open(module_registry_path, 'r') as f:
                module_registry = json.load(f)
            
            modules = module_registry.get("modules", {})
            
            # Check for content hash duplicates
            content_hashes = {}
            duplicates_found = False
            
            for module_name, module_data in modules.items():
                content_hash = module_data.get("content_hash")
                if content_hash:
                    if content_hash in content_hashes:
                        self.deletion_candidates.append(module_name)
                        duplicates_found = True
                        print(f"🔥 DUPLICATE DETECTED: {module_name} (hash: {content_hash})")
                    else:
                        content_hashes[content_hash] = module_name
            
            if duplicates_found:
                self.violations.append(f"Duplicates found: {len(self.deletion_candidates)} modules")
            
            return duplicates_found
            
        except Exception as e:
            self.violations.append(f"Duplicate scan failed: {str(e)}")
            return False
    
    def mark_as_deletion_candidate(self) -> None:
        """Mark duplicate files for deletion."""
        if self.deletion_candidates:
            print(f"🔥 Marking {len(self.deletion_candidates)} modules as deletion candidates")
            self.log_build_tracker(f"🔥 DELETION_CANDIDATES: {self.deletion_candidates}")
    
    def is_using_real_data(self) -> bool:
        """Verify all modules use real data only, no mock data."""
        print("📊 Verifying real data usage...")
        
        try:
            # Check live_data.json is empty
            live_data_path = os.path.join(self.workspace_path, "live_data.json")
            with open(live_data_path, 'r') as f:
                live_data = json.load(f)
            
            mock_sources = live_data.get("live_data_sources", {})
            if mock_sources:
                self.violations.append(f"Mock data sources detected: {list(mock_sources.keys())}")
                return False
            
            # Check real_data.json has valid sources
            real_data_path = os.path.join(self.workspace_path, "real_data.json")
            with open(real_data_path, 'r') as f:
                real_data = json.load(f)
            
            data_sources = real_data.get("data_sources", {})
            if not data_sources:
                self.violations.append("No real data sources configured")
                return False
            
            print("✅ Real data usage verified")
            return True
            
        except Exception as e:
            self.violations.append(f"Data usage validation failed: {str(e)}")
            return False
    
    def is_telemetry_active(self) -> bool:
        """Verify telemetry is active and reporting."""
        print("📡 Verifying telemetry status...")
        
        try:
            telemetry_path = os.path.join(self.workspace_path, "telemetry.json")
            with open(telemetry_path, 'r') as f:
                telemetry = json.load(f)
            
            real_time_monitoring = telemetry.get("real_time_monitoring", False)
            if not real_time_monitoring:
                self.violations.append("Real-time monitoring not active")
                return False
            
            metrics = telemetry.get("metrics", {})
            if not metrics:
                self.violations.append("No telemetry metrics configured")
                return False
            
            print("✅ Telemetry verification passed")
            return True
            
        except Exception as e:
            self.violations.append(f"Telemetry validation failed: {str(e)}")
            return False
    
    def update_build_status(self) -> None:
        """Update build_status.json with current compliance state."""
        try:
            build_status_path = os.path.join(self.workspace_path, "build_status.json")
            with open(build_status_path, 'r') as f:
                build_status = json.load(f)
            
            build_status.update({
                "architect_mode_v3_compliance": True,
                "compliance_check_timestamp": datetime.now().isoformat(),
                "violations_detected": len(self.violations),
                "repair_candidates": len(self.repair_needed),
                "deletion_candidates": len(self.deletion_candidates),
                "last_compliance_check": datetime.now().isoformat()
            })
            
            with open(build_status_path, 'w') as f:
                json.dump(build_status, f, indent=2)
            
            print("✅ Build status updated")
            
        except Exception as e:
            self.violations.append(f"Build status update failed: {str(e)}")
    
    def log_build_tracker(self, message: str) -> None:
        """Log actions to build_tracker.md."""
        try:
            tracker_path = os.path.join(self.workspace_path, "build_tracker.md")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = f"\n### ARCHITECT MODE v3.0 COMPLIANCE CHECK - {timestamp}\n\n"
            log_entry += f"STATUS **{message}**\n\n"
            
            if self.violations:
                log_entry += "VIOLATIONS **Found:**\n"
                for violation in self.violations:
                    log_entry += f"- {violation}\n"
                log_entry += "\n"
            
            if self.deletion_candidates:
                log_entry += f"🔥 **DELETION CANDIDATES:** {len(self.deletion_candidates)} modules\n"
                for candidate in self.deletion_candidates[:5]:  # Show first 5
                    log_entry += f"- {candidate}\n"
                log_entry += "\n"
            
            log_entry += "---\n"
            
            with open(tracker_path, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            print(f"📝 Build tracker updated: {message}")
            
        except Exception as e:
            print(f"⚠️ Build tracker update failed: {str(e)}")
    
    def handle_violation(self, violation_msg: str) -> None:
        """Handle architecture violations."""
        self.violations.append(violation_msg)
        self.log_build_tracker(f"🚨 VIOLATION DETECTED: {violation_msg}")
        print(f"🚨 VIOLATION: {violation_msg}")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        return {
            "architect_mode_version": "v3.0",
            "compliance_check_timestamp": datetime.now().isoformat(),
            "status": "COMPLIANT" if not self.violations else "VIOLATIONS_DETECTED",
            "violations_count": len(self.violations),
            "violations": self.violations,
            "repair_needed_count": len(self.repair_needed),
            "repair_needed": self.repair_needed,
            "deletion_candidates_count": len(self.deletion_candidates),
            "deletion_candidates": self.deletion_candidates[:10],  # Show first 10
            "core_files_status": "VALIDATED",
            "workspace_path": self.workspace_path
        }


def main():
    """🏁 ARCHITECT MODE v3.0 EXECUTION ENTRY POINT"""
    print("🔐 GENESIS AI AGENT — ARCHITECT LOCK-IN v3.0")
    print("🚨 STRICT ARCHITECT MODE IS NOW ACTIVE")
    
    try:
        engine = ArchitectV3ComplianceEngine()
        compliance_report = engine.enforce_architect_mode()
        
        print("\n" + "="*60)
        print("📊 ARCHITECT MODE v3.0 COMPLIANCE REPORT")
        print("="*60)
        print(f"Status: {compliance_report['status']}")
        print(f"Violations: {compliance_report['violations_count']}")
        print(f"Repair Needed: {compliance_report['repair_needed_count']}")
        print(f"Deletion Candidates: {compliance_report['deletion_candidates_count']}")
        print("="*60)
        
        if compliance_report['violations_count'] == 0:
            print("✅ GENESIS ARCHITECTURE COMPLIANCE VERIFIED")
        else:
            print("🚨 VIOLATIONS DETECTED - REPAIR REQUIRED")
            for violation in compliance_report['violations']:
                print(f"  - {violation}")
        
        return compliance_report
        
    except ArchitectModeViolation as e:
        print(f"🚨 ARCHITECT_LOCK_BROKEN: {str(e)}")
        print("🛑 HALTING ALL ACTIONS")
        sys.exit(1)
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


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
