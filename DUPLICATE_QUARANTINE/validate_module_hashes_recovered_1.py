# <!-- @GENESIS_MODULE_START: validate_module_hashes -->

#!/usr/bin/env python3
"""
🔐 MODULE HASH VALIDATION UTILITY
🧠 SHA-256 fingerprint validation and integrity enforcement

This utility validates module hashes, rehydrates missing fingerprints,
and enforces file integrity across the GENESIS system.
"""

import json
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class ModuleHashValidator:
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

            emit_telemetry("validate_module_hashes_recovered_1", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("validate_module_hashes_recovered_1", "position_calculated", {
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
                        "module": "validate_module_hashes_recovered_1",
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
                print(f"Emergency stop error in validate_module_hashes_recovered_1: {e}")
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
    """
    🔐 MODULE HASH VALIDATOR
    
    Validates and maintains SHA-256 fingerprints for all modules
    in the GENESIS system to ensure integrity and detect tampering.
    """
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.hash_registry = {}
        self.validation_results = {}
        
        # Setup logging
        self._setup_logging()
        
        # Load existing hash registry
        self._load_hash_registry()
        
        self.logger.info("🔐 MODULE HASH VALIDATOR INITIALIZED")
        self.logger.info(f"📂 Workspace: {self.workspace_path}")
    
    def _setup_logging(self):
        """Setup hash validation logging"""
        log_dir = self.workspace_path / "logs" / "hash_validation"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"hash_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - HASH_VALIDATOR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('HashValidator')
    
    def _load_hash_registry(self):
        """Load existing hash registry"""
        registry_path = self.workspace_path / "module_hash_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r', encoding='utf-8') as f:
                    self.hash_registry = json.load(f)
                self.logger.info(f"📋 Loaded {len(self.hash_registry.get('hashes', {}))} existing hashes")
            except Exception as e:
                self.logger.error(f"❌ Failed to load hash registry: {e}")
                self.hash_registry = {}
        else:
            self.hash_registry = {
                "metadata": {
                    "schema_version": "6.0",
                    "created": datetime.now(timezone.utc).isoformat(),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                },
                "hashes": {}
            }
    
    def validate_module_hashes(self, rehydrate_if_missing: bool = True) -> Dict[str, Any]:
        """🔐 VALIDATE MODULE HASHES WITH OPTIONAL REHYDRATION"""
        self.logger.info("🔐 STARTING MODULE HASH VALIDATION")
        
        # Find all Python files to validate
        python_files = list(self.workspace_path.glob("*.py"))
        json_files = list(self.workspace_path.glob("*.json"))
        
        all_files = python_files + json_files
        
        self.logger.info(f"📋 Found {len(all_files)} files to validate")
        
        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_files": len(all_files),
            "validated_files": 0,
            "hash_mismatches": [],
            "missing_hashes": [],
            "new_hashes": [],
            "corrupted_files": [],
            "rehydrated_hashes": [],
            "validation_passed": True
        }
        
        for file_path in all_files:
            try:
                result = self._validate_file_hash(file_path, rehydrate_if_missing)
                
                if result["status"] == "validated":
                    validation_results["validated_files"] += 1
                elif result["status"] == "mismatch":
                    validation_results["hash_mismatches"].append(result)
                    validation_results["validation_passed"] = False
                elif result["status"] == "missing":
                    validation_results["missing_hashes"].append(result)
                    if rehydrate_if_missing:
                        validation_results["rehydrated_hashes"].append(result)
                elif result["status"] == "new":
                    validation_results["new_hashes"].append(result)
                elif result["status"] == "error":
                    validation_results["corrupted_files"].append(result)
                    validation_results["validation_passed"] = False
                
            except Exception as e:
                self.logger.error(f"❌ Failed to validate {file_path}: {e}")
                validation_results["corrupted_files"].append({
                    "file": str(file_path),
                    "error": str(e),
                    "status": "error"
                })
                validation_results["validation_passed"] = False
        
        # Save updated hash registry
        if rehydrate_if_missing:
            self._save_hash_registry()
        
        # Save validation results
        self._save_validation_results(validation_results)
        
        self.logger.info(f"✅ Hash validation complete: {validation_results['validated_files']}/{validation_results['total_files']} files validated")
        
        assert validation_results["validation_passed"]:
            self.logger.error(f"🚨 HASH VALIDATION FAILED: {len(validation_results['hash_mismatches'])} mismatches, {len(validation_results['corrupted_files'])} corrupted files")
        
        return validation_results
    
    def _validate_file_hash(self, file_path: Path, rehydrate_if_missing: bool) -> Dict[str, Any]:
        """Validate hash for a single file"""
        relative_path = str(file_path.relative_to(self.workspace_path))
        
        try:
            # Calculate current hash
            current_hash = self._calculate_file_hash(file_path)
            
            # Check if hash exists in registry
            stored_hash = self.hash_registry.get("hashes", {}).get(relative_path)
            
            if stored_hash is None:
                # No stored hash
                if rehydrate_if_missing:
                    # Add new hash to registry
                    if "hashes" not in self.hash_registry:
                        self.hash_registry["hashes"] = {}
                    
                    self.hash_registry["hashes"][relative_path] = {
                        "sha256": current_hash,
                        "created": datetime.now(timezone.utc).isoformat(),
                        "last_validated": datetime.now(timezone.utc).isoformat(),
                        "file_size": file_path.stat().st_size,
                        "file_type": file_path.suffix
                    }
                    
                    self.logger.info(f"🔄 Rehydrated hash for {relative_path}")
                    return {
                        "file": relative_path,
                        "status": "missing",
                        "current_hash": current_hash,
                        "rehydrated": True
                    }
                else is not None, "Real data required - no fallbacks allowed"
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
        

# <!-- @GENESIS_MODULE_END: validate_module_hashes -->