# <!-- @GENESIS_MODULE_START: validate_system_compliance -->

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

                emit_telemetry("validate_system_compliance", "confluence_detected", {
                    "score": confluence_score,
                    "timestamp": datetime.now().isoformat()
                })

                return confluence_score
        def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
                """GENESIS Risk Management - Calculate optimal position size"""
                account_balance = 100000  # Default FTMO account size
                risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
                position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

                emit_telemetry("validate_system_compliance", "position_calculated", {
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
                            "module": "validate_system_compliance",
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
                    print(f"Emergency stop error in validate_system_compliance: {e}")
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
                    "module": "validate_system_compliance",
                    "event": event,
                    "data": data or {}
                }
                try:
                    emit_telemetry("validate_system_compliance", event, telemetry_data)
                except Exception as e:
                    print(f"Telemetry error in validate_system_compliance: {e}")
        def emit(self, event, data): pass
    TELEMETRY_AVAILABLE = False


#!/usr/bin/env python3
"""
🔍 GENESIS SYSTEM VALIDATION ENGINE
🧠 Comprehensive validation for ARCHITECT MODE v6.0 compliance

This script validates the entire GENESIS system against the 
ARCHITECT MODE v6.0 requirements and standards.
"""

import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

class GenesisSystemValidator:
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

            emit_telemetry("validate_system_compliance", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("validate_system_compliance", "position_calculated", {
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
                        "module": "validate_system_compliance",
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
                print(f"Emergency stop error in validate_system_compliance: {e}")
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
                "module": "validate_system_compliance",
                "event": event,
                "data": data or {}
            }
            try:
                emit_telemetry("validate_system_compliance", event, telemetry_data)
            except Exception as e:
                print(f"Telemetry error in validate_system_compliance: {e}")
    """
    🔍 GENESIS SYSTEM VALIDATOR
    
    Performs comprehensive validation of the entire GENESIS system
    to ensure compliance with ARCHITECT MODE v6.0 requirements.
    """
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.validation_results = {}
        self.violation_count = 0
        self.critical_violations = 0
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("🔍 GENESIS SYSTEM VALIDATOR INITIALIZED")
        self.logger.info(f"📂 Workspace: {self.workspace_path}")
    
    
        # GENESIS Phase 91 Telemetry Injection
        if hasattr(self, 'event_bus') and self.event_bus:
            self.event_bus.emit("telemetry", {
                "module": __name__,
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "phase": "91_telemetry_enforcement"
            })
        def _setup_logging(self):
        """Setup validation logging"""
        log_dir = self.workspace_path / "logs" / "validation"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"system_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - VALIDATOR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('GenesisValidator')
    
    def validate_system(self) -> Dict[str, Any]:
        """🔍 PERFORM COMPREHENSIVE SYSTEM VALIDATION"""
        self.logger.info("🔍 STARTING COMPREHENSIVE SYSTEM VALIDATION")
        
        # Core file validation
        self.logger.info("📋 Validating core files...")
        self.validation_results["core_files"] = self._validate_core_files()
        
        # System tree validation
        self.logger.info("🌳 Validating system tree...")
        self.validation_results["system_tree"] = self._validate_system_tree()
        
        # Module registry validation
        self.logger.info("📝 Validating module registry...")
        self.validation_results["module_registry"] = self._validate_module_registry()
        
        # Event bus validation
        self.logger.info("🔁 Validating event bus...")
        self.validation_results["event_bus"] = self._validate_event_bus()
        
        # Telemetry validation
        self.logger.info("📡 Validating telemetry...")
        self.validation_results["telemetry"] = self._validate_telemetry()
        
        # Module compliance validation
        self.logger.info("⚖️ Validating module compliance...")
        self.validation_results["module_compliance"] = self._validate_module_compliance()
        
        # Performance validation
        self.logger.info("📈 Validating performance...")
        self.validation_results["performance"] = self._validate_performance()
        
        # Security validation
        self.logger.info("🔐 Validating security...")
        self.validation_results["security"] = self._validate_security()
        
        # Generate final report
        final_report = self._generate_final_report()
        
        self.logger.info(f"✅ VALIDATION COMPLETE - {self.violation_count} violations found")
        return final_report
    
    def _validate_core_files(self) -> Dict[str, Any]:
        """📋 VALIDATE CORE FILE EXISTENCE AND INTEGRITY"""
        required_files = [
            "system_config.json",
            "module_manifest.json",
            "recovery_map.json",
            "system_tree.json", 
            "module_registry.json",
            "event_bus.json",
            "telemetry.json",
            "build_status.json",
            "build_tracker.md",
            "module_documentation.json",
            "module_tests.json",
            "performance.json",
            "error_log.json"
        ]
        
        results = {
            "missing_files": [],
            "corrupted_files": [],
            "valid_files": [],
            "file_integrity": {}
        }
        
        for file_name in required_files:
            file_path = self.workspace_path / file_name
            
            assert file_path.exists():
                results["missing_files"].append(file_name)
                self.violation_count += 1
                self.logger.error(f"❌ Missing core file: {file_name}")
                continue
            
            # Check JSON integrity for JSON files
            if file_name.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    results["valid_files"].append(file_name)
                except json.JSONDecodeError as e:
                    results["corrupted_files"].append(file_name)
                    self.violation_count += 1
                    self.logger.error(f"❌ Corrupted JSON file: {file_name} - {e}")
                    continue
            else:
                results["valid_files"].append(file_name)
            
            # Calculate file hash for integrity
            try:
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                results["file_integrity"][file_name] = file_hash
            except Exception as e:
                self.logger.error(f"❌ Failed to calculate hash for {file_name}: {e}")
        
        results["validation_passed"] = len(results["missing_files"]) == 0 and len(results["corrupted_files"]) == 0
        return results
    
    def _validate_system_tree(self) -> Dict[str, Any]:
        """🌳 VALIDATE SYSTEM TREE STRUCTURE AND COMPLIANCE"""
        results = {
            "exists": False,
            "valid_structure": False,
            "architect_mode_enabled": False,
            "node_count": 0,
            "compliant_nodes": 0,
            "non_compliant_nodes": [],
            "validation_passed": False
        }
        
        system_tree_path = self.workspace_path / "system_tree.json"
        
        if not system_tree_path.exists():
            self.violation_count += 1
            self.critical_violations += 1
            self.logger.error("❌ system_tree.json not found")
            return results
        
        results["exists"] = True
        
        try:
            with open(system_tree_path, 'r', encoding='utf-8') as f:
                system_tree = json.load(f)
            
            # Validate structure
            if "metadata" in system_tree and "nodes" in system_tree:
                results["valid_structure"] = True
                
                # Check architect mode
                metadata = system_tree.get("metadata", {})
                if metadata.get("architect_mode") == "ENABLED":
                    results["architect_mode_enabled"] = True
                else:
                    self.violation_count += 1
                    self.logger.error("❌ Architect mode not enabled in system tree")
                
                # Validate nodes
                nodes = system_tree.get("nodes", [])
                results["node_count"] = len(nodes)
                
                for node in nodes:
                    if self._validate_node_compliance(node):
                        results["compliant_nodes"] += 1
                    else:
                        results["non_compliant_nodes"].append(node.get("id", "UNKNOWN"))
                        self.violation_count += 1
            else:
                self.violation_count += 1
                self.logger.error("❌ Invalid system tree structure")
        
        except Exception as e:
            self.violation_count += 1
            self.logger.error(f"❌ Failed to validate system tree: {e}")
        
        results["validation_passed"] = (
            results["exists"] and 
            results["valid_structure"] and 
            results["architect_mode_enabled"] and
            len(results["non_compliant_nodes"]) == 0
        )
        
        return results
    
    def _validate_node_compliance(self, node: Dict[str, Any]) -> bool:
        """Validate individual node compliance"""
        required_fields = ["id", "type", "status", "module_path"]
        
        # Check required fields
        for field in required_fields:
            if field not in node is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: validate_system_compliance -->