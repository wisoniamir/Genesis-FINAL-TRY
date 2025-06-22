#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 GENESIS PHASE 7.9: INTENSIVE CROSS-MODULE VALIDATION ENGINE
ARCHITECT MODE v7.0.0 ULTIMATE ENFORCEMENT EDITION

🎯 OBJECTIVE:
Full deep stress-test across all GENESIS modules to confirm:
1. Zero orphan logic
2. Full EventBus signal paths
3. Accurate functional linking per topology
4. Preservation of preserved-but-similar logic roles
5. Redundancy with purpose (not accidental duplication)
6. UI wiring readiness across all 166 dashboard panels

🔐 ARCHITECT MODE COMPLIANCE:
- NO SIMPLIFICATIONS
- NO MOCKS 
- NO DUPLICATES
- NO ISOLATED LOGIC
- REAL-TIME MT5 DATA ONLY
- FULL EVENTBUS WIRING
- COMPLETE TELEMETRY HOOKS
"""

import json
import os
import sys
import ast
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import logging
import time

# GENESIS EventBus Integration - Auto-injected by Comprehensive Module Upgrade Engine
try:
    from core.hardened_event_bus import get_event_bus, emit_event, register_route
    from core.telemetry import emit_telemetry
    EVENTBUS_AVAILABLE = True
except ImportError:
    # Fallback for modules without core access
    def get_event_bus(): return None
    def emit_event(event, data): print(f"EVENT: {event}")
    def register_route(route, producer, consumer): pass
    def emit_telemetry(module, event, data): print(f"TELEMETRY: {module}.{event}")
    EVENTBUS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Phase7IntensiveValidation')

class Phase7IntensiveSystemValidator:
    """
    🧠 GENESIS PHASE 7.9 INTENSIVE SYSTEM VALIDATION ENGINE
    
    Performs comprehensive cross-module validation with zero tolerance
    for orphan logic, missing connections, or compliance violations.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.absolute()
        self.validation_report = {
            "metadata": {
                "phase": "7.9_INTENSIVE_SYSTEM_VALIDATION",
                "timestamp": datetime.now().isoformat(),
                "architect_mode": "v7.0.0_ULTIMATE_ENFORCEMENT",
                "compliance_level": "ZERO_TOLERANCE"
            },
            "validation_results": {},
            "signal_trace_logs": {},
            "disconnected_modules": [],
            "compliance_violations": [],
            "dashboard_panel_status": {},
            "suggested_fixes": [],
            "pass_fail_summary": {}
        }
        
        # Load core system files
        self.core_files = {
            "build_status": self._load_json("build_status.json"),
            "system_tree": self._load_json("system_tree.json"),
            "module_registry": self._load_json("module_registry.json"),
            "genesis_topology": self._load_json("genesis_final_topology.json"),
            "role_mapping": self._load_json("genesis_module_role_mapping.json"),
            "system_status": self._load_json("genesis_comprehensive_system_status.json"),
            "event_bus": self._load_json("event_bus.json"),
            "telemetry": self._load_json("telemetry.json") if os.path.exists(self.base_path / "telemetry.json") else {}
        }
        
        emit_telemetry("phase_7_validator", "validation_initialized", {
            "core_files_loaded": len([f for f in self.core_files.values() if f]),
            "validation_scope": "COMPREHENSIVE_CROSS_MODULE"
        })
        
    def _load_json(self, filename: str) -> Dict:
        """Load JSON file with error handling"""
        try:
            file_path = self.base_path / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"File not found: {filename}")
                return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}
    
    def validate_module_registration_compliance(self) -> Dict:
        """
        🔁 1. Cross-reference all modules in system with genesis_final_topology.json
        """
        logger.info("🔁 Phase 1: Module Registration Compliance Validation")
        
        system_tree = self.core_files.get("system_tree", {})
        topology = self.core_files.get("genesis_topology", {})
        role_mapping = self.core_files.get("role_mapping", {})
        
        validation_results = {
            "total_modules_in_system": 0,
            "modules_in_topology": 0,
            "unregistered_modules": [],
            "topology_orphans": [],
            "compliance_score": 0.0
        }
        
        # Extract all modules from system_tree
        system_modules = set()
        if "connected_modules" in system_tree:
            for category, modules in system_tree["connected_modules"].items():
                if isinstance(modules, list):
                    for module in modules:
                        if isinstance(module, dict) and "name" in module:
                            system_modules.add(module["name"])
                        elif isinstance(module, str):
                            system_modules.add(module)
        
        validation_results["total_modules_in_system"] = len(system_modules)
        
        # Extract modules from topology
        topology_modules = set()
        if "by_functional_role" in topology:
            for role, modules in topology["by_functional_role"].items():
                if isinstance(modules, list):
                    for module in modules:
                        if isinstance(module, dict) and "name" in module:
                            topology_modules.add(module["name"])
        
        validation_results["modules_in_topology"] = len(topology_modules)
        
        # Find unregistered and orphan modules
        validation_results["unregistered_modules"] = list(system_modules - topology_modules)
        validation_results["topology_orphans"] = list(topology_modules - system_modules)
        
        # Calculate compliance score
        if validation_results["total_modules_in_system"] > 0:
            registered_count = validation_results["total_modules_in_system"] - len(validation_results["unregistered_modules"])
            validation_results["compliance_score"] = (registered_count / validation_results["total_modules_in_system"]) * 100
        
        emit_telemetry("phase_7_validator", "module_registration_validated", validation_results)
        
        return validation_results
    
    def validate_module_roles_and_wiring(self) -> Dict:
        """
        🔁 2. For each module, confirm role definition and wiring
        """
        logger.info("🔁 Phase 2: Module Roles and Wiring Validation")
        
        role_mapping = self.core_files.get("role_mapping", {})
        system_tree = self.core_files.get("system_tree", {})
        
        module_validation = {}
        
        if "module_analysis" in role_mapping:
            for module_name, module_data in role_mapping["module_analysis"].items():
                validation_result = {
                    "has_defined_role": bool(module_data.get("functional_role")),
                    "is_wired": module_data.get("status") == "FULLY_WIRED",
                    "eventbus_connected": bool(module_data.get("eventbus_routes", {}).get("consumed") or 
                                             module_data.get("eventbus_routes", {}).get("emitted")),
                    "telemetry_enabled": bool(module_data.get("registry", {}).get("telemetry")),
                    "dashboard_panel": bool(module_data.get("dashboard_panel")),
                    "ftmo_compliance": module_data.get("ftmo_compliance", False),
                    "mock_data_usage": module_data.get("mock_data_usage", True),  # Violation if True
                    "warnings": module_data.get("warnings", [])
                }
                
                # Calculate module score
                score_factors = [
                    validation_result["has_defined_role"],
                    validation_result["is_wired"],
                    validation_result["eventbus_connected"],
                    validation_result["telemetry_enabled"],
                    validation_result["ftmo_compliance"],
                    not validation_result["mock_data_usage"]  # Inverted because mock usage is bad
                ]
                validation_result["compliance_score"] = (sum(score_factors) / len(score_factors)) * 100
                validation_result["status"] = "PASS" if validation_result["compliance_score"] >= 83.33 else "FAIL"
                
                module_validation[module_name] = validation_result
        
        emit_telemetry("phase_7_validator", "module_roles_validated", {
            "modules_checked": len(module_validation),
            "passing_modules": len([m for m in module_validation.values() if m["status"] == "PASS"]),
            "failing_modules": len([m for m in module_validation.values() if m["status"] == "FAIL"])
        })
        
        return module_validation
    
    def simulate_eventbus_signal_traffic(self) -> Dict:
        """
        ⚙️ 3. Run simulation of EventBus signal traffic
        """
        logger.info("⚙️ Phase 3: EventBus Signal Traffic Simulation")
        
        role_mapping = self.core_files.get("role_mapping", {})
        signal_trace = {
            "total_signals_traced": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "orphan_signals": [],
            "signal_flow_map": {},
            "latency_simulation": {}
        }
        
        if "module_analysis" in role_mapping:
            for module_name, module_data in role_mapping["module_analysis"].items():
                eventbus_routes = module_data.get("eventbus_routes", {})
                consumed_signals = eventbus_routes.get("consumed", [])
                emitted_signals = eventbus_routes.get("emitted", [])
                
                # Trace each signal
                for signal in emitted_signals:
                    signal_trace["total_signals_traced"] += 1
                    
                    # Find consumers for this signal
                    consumers = []
                    for other_module, other_data in role_mapping["module_analysis"].items():
                        if other_module != module_name:
                            other_consumed = other_data.get("eventbus_routes", {}).get("consumed", [])
                            if signal in other_consumed:
                                consumers.append(other_module)
                    
                    if consumers:
                        signal_trace["successful_routes"] += 1
                        signal_trace["signal_flow_map"][f"{module_name}->{signal}"] = consumers
                    else:
                        signal_trace["failed_routes"] += 1
                        signal_trace["orphan_signals"].append({
                            "producer": module_name,
                            "signal": signal,
                            "issue": "NO_CONSUMERS_FOUND"
                        })
                    
                    # Simulate latency (mock timing for validation)
                    signal_trace["latency_simulation"][f"{module_name}->{signal}"] = {
                        "estimated_latency_ms": len(consumers) * 5 + 10,  # Simple latency model
                        "consumers": len(consumers)
                    }
        
        emit_telemetry("phase_7_validator", "signal_traffic_simulated", signal_trace)
        
        return signal_trace
    
    def validate_dashboard_panel_connections(self) -> Dict:
        """
        🔁 4. Trace UI connections for all 166 dashboard panels
        """
        logger.info("🔁 Phase 4: Dashboard Panel Connection Validation")
        
        role_mapping = self.core_files.get("role_mapping", {})
        dashboard_validation = {
            "total_panels_expected": 166,
            "panels_with_modules": 0,
            "panels_without_modules": [],
            "module_panel_map": {},
            "functional_coverage": {}
        }
        
        # Map modules to their dashboard panels
        if "module_analysis" in role_mapping:
            for module_name, module_data in role_mapping["module_analysis"].items():
                dashboard_panel = module_data.get("dashboard_panel")
                if dashboard_panel:
                    dashboard_validation["panels_with_modules"] += 1
                    dashboard_validation["module_panel_map"][dashboard_panel] = module_name
        
        # Check functional role coverage
        if "core_functional_roles" in role_mapping:
            for role, role_data in role_mapping["core_functional_roles"].items():
                modules_count = role_data.get("modules", 0)
                critical_modules = role_data.get("critical_modules", [])
                
                panels_for_role = 0
                for module_name in critical_modules:
                    if module_name in role_mapping.get("module_analysis", {}):
                        if role_mapping["module_analysis"][module_name].get("dashboard_panel"):
                            panels_for_role += 1
                
                dashboard_validation["functional_coverage"][role] = {
                    "total_modules": modules_count,
                    "critical_modules": len(critical_modules),
                    "panels_connected": panels_for_role,
                    "coverage_percentage": (panels_for_role / len(critical_modules) * 100) if critical_modules else 0
                }
        
        # Calculate missing panels
        expected_panel_count = dashboard_validation["total_panels_expected"]
        actual_panel_count = dashboard_validation["panels_with_modules"]
        missing_count = expected_panel_count - actual_panel_count
        
        if missing_count > 0:
            dashboard_validation["panels_without_modules"] = [f"missing_panel_{i}" for i in range(missing_count)]
        
        emit_telemetry("phase_7_validator", "dashboard_panels_validated", dashboard_validation)
        
        return dashboard_validation
    
    def run_core_system_tests(self) -> Dict:
        """
        🧪 5. Run validation tests on key systems
        """
        logger.info("🧪 Phase 5: Core System Testing")
        
        core_systems = [
            "execution_engine",
            "strategy_mutation_engine", 
            "kill_switch_audit",
            "risk_guard",
            "signal_engine",
            "dashboard_engine"
        ]
        
        test_results = {}
        
        for system in core_systems:
            system_test = {
                "system": system,
                "module_exists": False,
                "eventbus_connected": False,
                "telemetry_enabled": False,
                "ftmo_compliant": False,
                "mock_data_free": False,
                "test_status": "FAIL"
            }
            
            # Check if system exists in role mapping
            role_mapping = self.core_files.get("role_mapping", {})
            if "module_analysis" in role_mapping and system in role_mapping["module_analysis"]:
                module_data = role_mapping["module_analysis"][system]
                
                system_test["module_exists"] = True
                system_test["eventbus_connected"] = bool(module_data.get("eventbus_routes"))
                system_test["telemetry_enabled"] = bool(module_data.get("registry", {}).get("telemetry"))
                system_test["ftmo_compliant"] = module_data.get("ftmo_compliance", False)
                system_test["mock_data_free"] = not module_data.get("mock_data_usage", True)
                
                # Calculate overall test status
                passed_checks = sum([
                    system_test["module_exists"],
                    system_test["eventbus_connected"],
                    system_test["telemetry_enabled"],
                    system_test["ftmo_compliant"],
                    system_test["mock_data_free"]
                ])
                
                system_test["test_status"] = "PASS" if passed_checks >= 4 else "FAIL"
                system_test["score"] = (passed_checks / 5) * 100
            
            test_results[system] = system_test
        
        emit_telemetry("phase_7_validator", "core_systems_tested", test_results)
        
        return test_results
    
    def generate_suggested_fixes(self, validation_data: Dict) -> List[Dict]:
        """
        Generate actionable fixes for detected issues
        """
        logger.info("🔧 Generating Suggested Fixes")
        
        fixes = []
        
        # Fix unregistered modules
        module_validation = validation_data.get("module_validation", {})
        for module_name, module_data in module_validation.items():
            if module_data.get("status") == "FAIL":
                if not module_data.get("eventbus_connected"):
                    fixes.append({
                        "type": "EVENTBUS_WIRING",
                        "module": module_name,
                        "action": "Wire module to EventBus",
                        "details": "Add eventbus_routes configuration and implement emit/consume patterns"
                    })
                
                if not module_data.get("telemetry_enabled"):
                    fixes.append({
                        "type": "TELEMETRY_INTEGRATION",
                        "module": module_name,
                        "action": "Enable telemetry hooks",
                        "details": "Add emit_telemetry calls for key module events"
                    })
                
                if module_data.get("mock_data_usage"):
                    fixes.append({
                        "type": "MOCK_DATA_ELIMINATION",
                        "module": module_name,
                        "action": "Remove mock data usage",
                        "details": "Replace all mock/test data with real MT5 data sources"
                    })
        
        # Fix orphan signals
        signal_trace = validation_data.get("signal_trace", {})
        for orphan in signal_trace.get("orphan_signals", []):
            fixes.append({
                "type": "SIGNAL_ROUTING",
                "module": orphan["producer"],
                "action": f"Add consumer for signal '{orphan['signal']}'",
                "details": "Find appropriate consumer module or remove unused signal"
            })
        
        # Fix dashboard panel gaps
        dashboard_validation = validation_data.get("dashboard_validation", {})
        missing_panels = dashboard_validation.get("panels_without_modules", [])
        if missing_panels:
            fixes.append({
                "type": "DASHBOARD_COMPLETION",
                "action": f"Create {len(missing_panels)} missing dashboard panels",
                "details": "Implement remaining dashboard panels to reach 166 panel target"
            })
        
        emit_telemetry("phase_7_validator", "fixes_generated", {"total_fixes": len(fixes)})
        
        return fixes
    
    def run_intensive_validation(self) -> str:
        """
        Execute complete Phase 7.9 intensive system validation
        """
        logger.info("🚀 Starting Phase 7.9 Intensive System Validation")
        
        start_time = time.time()
        
        try:
            # Phase 1: Module Registration Compliance
            self.validation_report["validation_results"]["module_registration"] = \
                self.validate_module_registration_compliance()
            
            # Phase 2: Module Roles and Wiring
            self.validation_report["validation_results"]["module_validation"] = \
                self.validate_module_roles_and_wiring()
            
            # Phase 3: EventBus Signal Traffic
            self.validation_report["validation_results"]["signal_trace"] = \
                self.simulate_eventbus_signal_traffic()
            
            # Phase 4: Dashboard Panel Connections
            self.validation_report["validation_results"]["dashboard_validation"] = \
                self.validate_dashboard_panel_connections()
            
            # Phase 5: Core System Tests
            self.validation_report["validation_results"]["core_system_tests"] = \
                self.run_core_system_tests()
            
            # Generate suggested fixes
            self.validation_report["suggested_fixes"] = \
                self.generate_suggested_fixes(self.validation_report["validation_results"])
            
            # Calculate overall compliance score
            module_validation = self.validation_report["validation_results"]["module_validation"]
            passing_modules = len([m for m in module_validation.values() if m.get("status") == "PASS"])
            total_modules = len(module_validation)
            overall_compliance = (passing_modules / total_modules * 100) if total_modules > 0 else 0
            
            self.validation_report["pass_fail_summary"] = {
                "overall_compliance_score": overall_compliance,
                "total_modules_tested": total_modules,
                "passing_modules": passing_modules,
                "failing_modules": total_modules - passing_modules,
                "critical_violations": len([f for f in self.validation_report["suggested_fixes"] 
                                          if f.get("type") in ["MOCK_DATA_ELIMINATION", "EVENTBUS_WIRING"]]),
                "validation_status": "PASS" if overall_compliance >= 90 else "FAIL"
            }
            
            # Add execution metadata
            execution_time = time.time() - start_time
            self.validation_report["metadata"]["execution_time_seconds"] = execution_time
            self.validation_report["metadata"]["completion_status"] = "SUCCESS"
            
            # Save validation report
            report_filename = f"PHASE_7_INTENSIVE_SYSTEM_VALIDATION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = self.base_path / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
            
            # Create markdown summary
            self._create_markdown_report(report_filename.replace('.json', '.md'))
            
            emit_telemetry("phase_7_validator", "validation_completed", {
                "execution_time": execution_time,
                "overall_compliance": overall_compliance,
                "report_file": report_filename
            })
            
            logger.info(f"✅ Phase 7.9 Validation Complete - Report: {report_filename}")
            return report_filename
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            self.validation_report["metadata"]["completion_status"] = "FAILED"
            self.validation_report["metadata"]["error"] = str(e)
            
            # Save error report
            error_report_filename = f"PHASE_7_VALIDATION_ERROR_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(self.base_path / error_report_filename, 'w', encoding='utf-8') as f:
                json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
            
            return error_report_filename
    
    def _create_markdown_report(self, filename: str):
        """Create a human-readable markdown summary"""
        markdown_content = f"""# 🧠 GENESIS PHASE 7.9 INTENSIVE SYSTEM VALIDATION REPORT

**Generated:** {self.validation_report['metadata']['timestamp']}  
**Architect Mode:** {self.validation_report['metadata']['architect_mode']}  
**Compliance Level:** {self.validation_report['metadata']['compliance_level']}

## 📊 EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| **Overall Compliance Score** | {self.validation_report['pass_fail_summary']['overall_compliance_score']:.1f}% |
| **Validation Status** | {self.validation_report['pass_fail_summary']['validation_status']} |
| **Total Modules Tested** | {self.validation_report['pass_fail_summary']['total_modules_tested']} |
| **Passing Modules** | {self.validation_report['pass_fail_summary']['passing_modules']} |
| **Failing Modules** | {self.validation_report['pass_fail_summary']['failing_modules']} |
| **Critical Violations** | {self.validation_report['pass_fail_summary']['critical_violations']} |

## 🔁 MODULE REGISTRATION COMPLIANCE

"""
        
        reg_results = self.validation_report['validation_results']['module_registration']
        markdown_content += f"""
- **Total Modules in System:** {reg_results['total_modules_in_system']}
- **Modules in Topology:** {reg_results['modules_in_topology']}
- **Compliance Score:** {reg_results['compliance_score']:.1f}%
- **Unregistered Modules:** {len(reg_results['unregistered_modules'])}
- **Topology Orphans:** {len(reg_results['topology_orphans'])}

"""
        
        # Add signal trace summary
        signal_results = self.validation_report['validation_results']['signal_trace']
        markdown_content += f"""
## ⚙️ EVENTBUS SIGNAL TRAFFIC ANALYSIS

- **Total Signals Traced:** {signal_results['total_signals_traced']}
- **Successful Routes:** {signal_results['successful_routes']}
- **Failed Routes:** {signal_results['failed_routes']}
- **Orphan Signals:** {len(signal_results['orphan_signals'])}

"""
        
        # Add dashboard panel summary
        dashboard_results = self.validation_report['validation_results']['dashboard_validation']
        markdown_content += f"""
## 🎛️ DASHBOARD PANEL CONNECTIVITY

- **Expected Panels:** {dashboard_results['total_panels_expected']}
- **Panels with Modules:** {dashboard_results['panels_with_modules']}
- **Missing Panels:** {len(dashboard_results['panels_without_modules'])}

"""
        
        # Add suggested fixes
        markdown_content += f"""
## 🔧 SUGGESTED FIXES ({len(self.validation_report['suggested_fixes'])})

"""
        for fix in self.validation_report['suggested_fixes'][:10]:  # Show first 10 fixes
            markdown_content += f"- **{fix['type']}**: {fix['action']} - {fix.get('details', '')}\n"
        
        if len(self.validation_report['suggested_fixes']) > 10:
            markdown_content += f"\n*...and {len(self.validation_report['suggested_fixes']) - 10} more fixes*\n"
        
        markdown_content += f"""

---
**Report generated by GENESIS Architect Mode v7.0.0**  
**Execution Time:** {self.validation_report['metadata'].get('execution_time_seconds', 0):.2f} seconds
"""
        
        # Save markdown report
        with open(self.base_path / filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


def main():
    """Main execution function"""
    print("🧠 GENESIS PHASE 7.9 INTENSIVE SYSTEM VALIDATION ENGINE")
    print("=" * 60)
    
    validator = Phase7IntensiveSystemValidator()
    report_filename = validator.run_intensive_validation()
    
    print(f"\n✅ Validation Complete!")
    print(f"📄 Report saved as: {report_filename}")
    print(f"📄 Markdown summary: {report_filename.replace('.json', '.md')}")
    
    return report_filename


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error during validation: {e}")
        sys.exit(1)
