import logging
#!/usr/bin/env python3
"""
🧾 GENESIS BUILD CONTINUITY GUARD v1.0 — Phase Memory & Progress Control

🎯 OBJECTIVE: Prevent redundant builds, ensure surgical enhancements only,
and maintain strict build continuity with complete phase memory.

🔒 ARCHITECT MODE COMPLIANCE:
- ✅ Operates under system_mode: "quarantined_compliance_enforcer"
- ✅ Enforces build continuity rules
- ✅ Prevents redundant module creation
- ✅ Ensures surgical enhancement approach only
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

class GenesisBuildContinuityGuard:
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

            emit_telemetry("build_continuity_guard", "confluence_detected", {
                "score": confluence_score,
                "timestamp": datetime.now().isoformat()
            })

            return confluence_score
    def emergency_stop(self, reason: str = "Manual trigger") -> bool:
            """GENESIS Emergency Kill Switch"""
            emit_event("emergency_stop", {
                "module": "build_continuity_guard",
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            emit_telemetry("build_continuity_guard", "kill_switch_activated", {
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })

            return True
    def calculate_position_size(self, risk_amount: float, stop_loss_pips: float) -> float:
            """GENESIS Risk Management - Calculate optimal position size"""
            account_balance = 100000  # Default FTMO account size
            risk_per_pip = risk_amount / stop_loss_pips if stop_loss_pips > 0 else 0
            position_size = min(risk_per_pip * 0.01, account_balance * 0.02)  # Max 2% risk

            emit_telemetry("build_continuity_guard", "position_calculated", {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percentage": (position_size / account_balance) * 100
            })

            return position_size
    def validate_ftmo_compliance(self, trade_data: dict) -> bool:
            """GENESIS FTMO Compliance Validator"""
            # Daily drawdown check (5%)
            daily_loss = trade_data.get('daily_loss', 0)
            if daily_loss > 0.05:
                emit_telemetry("build_continuity_guard", "ftmo_violation", {"type": "daily_drawdown", "value": daily_loss})
                return False

            # Maximum drawdown check (10%)
            max_drawdown = trade_data.get('max_drawdown', 0)
            if max_drawdown > 0.10:
                emit_telemetry("build_continuity_guard", "ftmo_violation", {"type": "max_drawdown", "value": max_drawdown})
                return False

            return True
    def log_state(self):
        """GENESIS Telemetry Enforcer - Log current module state"""
        state_data = {
            "module": "build_continuity_guard",
            "timestamp": datetime.now().isoformat(),
            "status": "active",
            "compliance_enforced": True
        }
        if hasattr(self, 'event_bus') and self.event_bus:
            emit_telemetry("build_continuity_guard", "state_update", state_data)
        return state_data

    """🧾 Build continuity enforcement and phase memory system"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.build_tracker_path = self.workspace_path / "build_tracker.md"
        self.build_status_path = self.workspace_path / "build_status.json"
        self.system_tree_path = self.workspace_path / "system_tree.json"
        self.patch_registry_path = self.workspace_path / "patch_registry.json"
        
        # Load current system state
        self.build_status = self.load_build_status()
        self.system_tree = self.load_system_tree()
        self.patch_registry = self.load_patch_registry()
        self.build_history = self.analyze_build_history()
        
        # Build continuity status
        self.continuity_status = {
            "timestamp": datetime.now().isoformat(),
            "guard_version": "v1.0",
            "system_mode": self.build_status.get("system_mode", "UNKNOWN"),
            "last_phase_completed": self.extract_last_completed_phase(),
            "modules_exist_and_complete": self.count_existing_modules(),
            "build_continuity": "UNKNOWN"
        }
    
    def load_build_status(self) -> Dict[str, Any]:
        """📋 Load build status JSON"""
        try:
            if self.build_status_path.exists():
                with open(self.build_status_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"status": "FILE_NOT_FOUND"}
        except Exception as e:
            return {"status": "LOAD_ERROR", "error": str(e)}
    
    def load_system_tree(self) -> Dict[str, Any]:
        """🌲 Load system tree JSON"""
        try:
            if self.system_tree_path.exists():
                with open(self.system_tree_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"status": "FILE_NOT_FOUND"}
        except Exception as e:
            return {"status": "LOAD_ERROR", "error": str(e)}
    
    def load_patch_registry(self) -> Dict[str, Any]:
        """📊 Load patch registry if in production mode"""
        try:
            if self.patch_registry_path.exists():
                with open(self.patch_registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {"status": "FILE_NOT_FOUND", "mode": "development"}
        except Exception as e:
            return {"status": "LOAD_ERROR", "error": str(e)}
    
    def analyze_build_history(self) -> Dict[str, Any]:
        """📋 Analyze build tracker for phase history"""
        build_history = {
            "phases_completed": [],
            "modules_created": [],
            "last_major_action": "UNKNOWN",
            "recovery_operations": [],
            "triage_operations": []
        }
        
        if self.build_tracker_path.exists():
            try:
                with open(self.build_tracker_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract completed phases
                import re


# <!-- @GENESIS_MODULE_END: build_continuity_guard -->


# <!-- @GENESIS_MODULE_START: build_continuity_guard -->
                phase_patterns = [
                    r'PHASE (\\d+).*?(?:COMPLETED|DEPLOYED|ACTIVE)',
                    r'Phase (\\d+).*?(?:completed|deployed|active)',
                    r'phase_(\\d+).*?(?:status.*?DEPLOYED|COMPLETED)'
                ]
                
                for pattern in phase_patterns:
                    phases = re.findall(pattern, content, re.IGNORECASE)
                    build_history["phases_completed"].extend([int(p) for p in phases])
                
                # Remove duplicates and sort
                build_history["phases_completed"] = sorted(list(set(build_history["phases_completed"])))
                
                # Extract major operations
                if "MODULE RECOVERY ENGINE" in content:
                    build_history["recovery_operations"].append("Module Recovery Engine Executed")
                if "EMERGENCY TRIAGE" in content:
                    build_history["triage_operations"].append("Emergency Triage Completed")
                if "ORPHAN INTENT CLASSIFICATION" in content:
                    build_history["recovery_operations"].append("Orphan Intent Classification Completed")
                if "ARCHITECTURE STATUS" in content:
                    build_history["last_major_action"] = "Architecture Status Generation"
                
            except Exception as e:
                build_history["error"] = str(e)
        
        return build_history
    
    def extract_last_completed_phase(self) -> int:
        """📊 Extract the last completed phase from system data"""
        # Check build status first
        if "phase_101_status" in self.build_status and self.build_status["phase_101_status"] == "DEPLOYED":
            return 101
        
        # Check build history
        if self.build_history["phases_completed"]:
            return max(self.build_history["phases_completed"])
        
        # Default fallback
        return 100
    
    def count_existing_modules(self) -> int:
        """📊 Count existing active modules"""
        return self.build_status.get("active_modules", 0)
    
    def check_module_exists(self, module_name: str) -> Dict[str, Any]:
        """🔍 Check if module already exists and its status"""
        module_info = {
            "exists": False,
            "status": "NOT_FOUND",
            "location": None,
            "completion_level": "UNKNOWN",
            "requires_action": "CREATE"
        }
        
        # Check if file exists in workspace
        possible_paths = [
            self.workspace_path / f"{module_name}.py",
            self.workspace_path / f"{module_name}",
            self.workspace_path / "src" / f"{module_name}.py",
            self.workspace_path / "core" / f"{module_name}.py"
        ]
        
        for path in possible_paths:
            if path.exists():
                module_info["exists"] = True
                module_info["location"] = str(path)
                module_info["status"] = "EXISTS"
                
                # Analyze completion level
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for completion indicators
                    if "production_ready" in content.lower():
                        module_info["completion_level"] = "PRODUCTION_READY"
                        module_info["requires_action"] = "SKIP"
                    elif "eventbus" in content.lower() and "telemetry" in content.lower():
                        module_info["completion_level"] = "PARTIALLY_COMPLETE"
                        module_info["requires_action"] = "ENHANCE"
                    elif len(content) > 1000:
                        module_info["completion_level"] = "SUBSTANTIAL"
                        module_info["requires_action"] = "ENHANCE"
                    else:
                        module_info["completion_level"] = "STUB"
                        module_info["requires_action"] = "ENHANCE"
                        
                except Exception:
                    module_info["completion_level"] = "UNKNOWN"
                    module_info["requires_action"] = "REVIEW"
                break
        
        return module_info
    
    def assess_next_intelligent_action(self, requested_action: str, target_module: Optional[str] = None) -> Dict[str, Any]:
        """🧠 Determine intelligent next action based on system state"""
        assessment = {
            "timestamp": datetime.now().isoformat(),
            "requested_action": requested_action,
            "target_module": target_module,
            "recommended_action": "UNKNOWN",
            "reasoning": [],
            "preconditions_met": False,
            "should_proceed": False
        }
        
        # Check system mode restrictions
        allowed_actions = self.build_status.get("allowed_actions", [])
        if requested_action.lower() not in [action.lower() for action in allowed_actions]:
            assessment["reasoning"].append(f"Action '{requested_action}' not allowed in current system mode")
            assessment["recommended_action"] = "WAIT_FOR_MODE_CHANGE"
            return assessment
        
        # Check if module already exists
        if target_module:
            module_status = self.check_module_exists(target_module)
            assessment["module_status"] = module_status
            
            if module_status["exists"]:
                if module_status["completion_level"] == "PRODUCTION_READY":
                    assessment["reasoning"].append(f"Module {target_module} already production-ready")
                    assessment["recommended_action"] = "SKIP"
                elif module_status["completion_level"] in ["PARTIALLY_COMPLETE", "SUBSTANTIAL"]:
                    assessment["reasoning"].append(f"Module {target_module} exists but incomplete")
                    assessment["recommended_action"] = "ENHANCE"
                    assessment["should_proceed"] = True
                    assessment["preconditions_met"] = True
                else:
                    assessment["reasoning"].append(f"Module {target_module} exists as stub")
                    assessment["recommended_action"] = "ENHANCE"
                    assessment["should_proceed"] = True
                    assessment["preconditions_met"] = True
            else:
                assessment["reasoning"].append(f"Module {target_module} does not exist")
                assessment["recommended_action"] = "CREATE"
                assessment["should_proceed"] = True
                assessment["preconditions_met"] = True
        
        # Check build continuity
        orphan_count = self.build_status.get("orphans_after_triage", 0)
        if orphan_count > 1000:
            assessment["reasoning"].append(f"Too many orphans ({orphan_count}) - triage required first")
            assessment["recommended_action"] = "TRIAGE_FIRST"
            assessment["should_proceed"] = False
        
        duplicate_count = self.system_tree.get("genesis_system_metadata", {}).get("duplicate_candidates", [])
        if isinstance(duplicate_count, list) and len(duplicate_count) > 100:
            assessment["reasoning"].append(f"Too many duplicates ({len(duplicate_count)}) - cleanup required first")
            assessment["recommended_action"] = "CLEANUP_DUPLICATES_FIRST"
        
        return assessment
    
    def emit_build_continuity_status(self) -> str:
        """📊 Generate build continuity status block"""
        status_block = f"""
🧠 AGENT BUILD CONTINUITY STATUS
- last_completed_phase: {self.continuity_status['last_phase_completed']}
- system_mode: {self.continuity_status['system_mode']}
- modules_exist_and_complete: {self.continuity_status['modules_exist_and_complete']}
- triage_status: {self.build_status.get('structural_compliance', 'UNKNOWN')}
- orphan_count: {self.build_status.get('orphans_after_triage', 'UNKNOWN')}
- recovery_status: {self.build_status.get('module_recovery_completed', False)}
- next_recommended_action: {self.build_status.get('next_action_required', 'Manual syntax review for 6 execution modules')}
- build_continuity: {self.assess_overall_continuity()}
"""
        return status_block
    
    def assess_overall_continuity(self) -> str:
        """📊 Assess overall build continuity health"""
        factors = []
        
        # Check if core operations completed
        if self.build_status.get("triage_completed", False):
            factors.append("triage_completed")
        
        if self.build_status.get("module_recovery_completed", False):
            factors.append("recovery_completed")
        
        if self.build_status.get("architect_mode_v3", False):
            factors.append("architect_mode_active")
        
        if self.build_status.get("orphans_after_triage", 0) < 1000:
            factors.append("orphan_count_acceptable")
        
        # Determine continuity level
        if len(factors) >= 3:
            return "EXCELLENT"
        elif len(factors) >= 2:
            return "GOOD"
        elif len(factors) >= 1:
            return "FAIR"
        else:
            return "POOR"
    
    def enforce_build_continuity_rules(self, proposed_action: str, target_module: Optional[str] = None) -> Dict[str, Any]:
        """🔒 Enforce build continuity rules before any action"""
        print("🧾 GENESIS BUILD CONTINUITY GUARD — ENFORCING PHASE MEMORY")
        print("=" * 60)
        
        # Load and validate core files
        print("📋 Loading core system files...")
        validation_results = {
            "build_tracker.md": self.build_tracker_path.exists(),
            "build_status.json": self.build_status_path.exists(),
            "system_tree.json": self.system_tree_path.exists(),
            "patch_registry.json": self.patch_registry_path.exists()
        }
        
        print("🔍 Analyzing system state...")
        
        # Emit current status
        status_block = self.emit_build_continuity_status()
        print(status_block)
        
        # Assess proposed action
        assessment = self.assess_next_intelligent_action(proposed_action, target_module)
        
        # Build enforcement decision
        enforcement_decision = {
            "timestamp": datetime.now().isoformat(),
            "guard_active": True,
            "core_files_validation": validation_results,
            "continuity_status": self.continuity_status,
            "build_history": self.build_history,
            "action_assessment": assessment,
            "enforcement_result": {
                "action_allowed": assessment["should_proceed"],
                "recommended_action": assessment["recommended_action"],
                "reasoning": assessment["reasoning"],
                "next_steps": []
            }
        }
        
        # Generate recommendations
        if assessment["should_proceed"]:
            enforcement_decision["enforcement_result"]["next_steps"] = [
                "Proceed with surgical enhancement approach",
                "Log all actions in build_tracker.md",
                "Update build_status.json after completion",
                "Maintain EventBus connectivity compliance"
            ]
            print(f"✅ ACTION APPROVED: {assessment['recommended_action']}")
        else:
            enforcement_decision["enforcement_result"]["next_steps"] = [
                f"Complete prerequisite: {assessment['recommended_action']}",
                "Re-run build continuity check",
                "Address system mode restrictions",
                "Ensure compliance with architect mode"
            ]
            print(f"🚫 ACTION BLOCKED: {assessment['recommended_action']} required first")
        
        return enforcement_decision
    
    def save_continuity_log(self, enforcement_decision: Dict[str, Any]):
        """📝 Save continuity enforcement log"""
        log_path = self.workspace_path / "build_continuity_log.json"
        
        try:
            # Load existing log or create new
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            else:
                log_data = {"continuity_log": []}
            
            # Add new entry
            log_data["continuity_log"].append(enforcement_decision)
            
            # Save updated log
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
                
            print(f"📝 Continuity log saved to: {log_path}")
            
        except Exception as e:
            print(f"❌ Failed to save continuity log: {str(e)}")

def enforce_build_continuity(proposed_action: str, target_module: Optional[str] = None) -> Dict[str, Any]:
    """🚀 Main build continuity enforcement entry point"""
    workspace_path = r"c:\Users\patra\Genesis FINAL TRY"
    
    guard = GenesisBuildContinuityGuard(workspace_path)
    enforcement_decision = guard.enforce_build_continuity_rules(proposed_action, target_module)
    guard.save_continuity_log(enforcement_decision)
    
    return enforcement_decision

def main():
    """🚀 Main execution for testing"""
    # Example enforcement check
    result = enforce_build_continuity("create", "test_module")
    
    print(f"\\n🎯 BUILD CONTINUITY ENFORCEMENT RESULT:")
    print(f"   Action Allowed: {result['enforcement_result']['action_allowed']}")
    print(f"   Recommended: {result['enforcement_result']['recommended_action']}")
    print(f"   Reasoning: {', '.join(result['action_assessment']['reasoning'])}")

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
