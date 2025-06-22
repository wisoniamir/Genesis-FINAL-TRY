#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 GENESIS PHASE 7.95: TRADING INTELLIGENCE AUDIT ENGINE
ARCHITECT MODE v7.0.0 ULTIMATE ENFORCEMENT EDITION

🧠 OBJECTIVE:
Deep audit of GENESIS trading intelligence system to validate:
1. Core strategy modules linked logically and temporally
2. Execution flow: macro → confluence → decision → risk → audit
3. Active, connected, signal-aware logic blocks (NO STUBS)
4. EventBus routes between trading-critical modules
5. Full stack trace validation

🔐 ARCHITECT MODE COMPLIANCE:
- NO SIMPLIFICATIONS
- NO MOCKS 
- NO STUBS
- NO ISOLATED LOGIC
- REAL-TIME MT5 DATA ONLY
- FULL EVENTBUS WIRING
- COMPLETE TELEMETRY HOOKS
"""

import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import ast

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
logger = logging.getLogger('TradingIntelligenceAudit')

class TradingIntelligenceAuditEngine:
    """
    🎯 GENESIS PHASE 7.95 TRADING INTELLIGENCE AUDIT ENGINE
    
    Performs comprehensive audit of trading intelligence system with zero tolerance
    for stubs, placeholders, or disconnected trading logic.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.absolute()
        self.audit_report = {
            "metadata": {
                "phase": "7.95_TRADING_INTELLIGENCE_AUDIT",
                "timestamp": datetime.now().isoformat(),
                "architect_mode": "v7.0.0_ULTIMATE_ENFORCEMENT",
                "compliance_level": "ZERO_TOLERANCE_TRADING_FOCUS"
            },
            "critical_modules_analysis": {},
            "trading_flow_validation": {},
            "signal_route_trace": {},
            "execution_path_tests": {},
            "intelligence_gaps": [],
            "mock_stub_violations": [],
            "recommendations": [],
            "trading_intelligence_score": 0.0
        }
        
        # Define critical trading modules to audit
        self.critical_trading_modules = {
            "execution_engine": [
                "execution_engine.py",
                "modules/execution/execution_engine.py"
            ],
            "strategy_mutation_engine": [
                "strategy_mutation_engine.py"
            ],
            "pattern_classifier_engine": [
                "modules/ml/pattern_classifier_engine.py"
            ],
            "macro_sync_engine": [
                "modules/restored/macro_sync_engine.py"
            ],
            "kill_switch_audit": [
                "kill_switch_audit.py"
            ],
            "risk_guard": [
                "risk_guard.py"
            ],
            "trade_journal_logger": [
                "trade_journal_logger.py",
                "modules/logging/trade_journal_logger.py"
            ],
            "performance_feedback_loop": [
                "performance_feedback_loop.py",
                "modules/feedback/performance_feedback_loop.py"
            ]
        }
        
        # Load core system files
        self.core_files = {
            "system_tree": self._load_json("system_tree.json"),
            "topology": self._load_json("genesis_final_topology.json"),
            "role_mapping": self._load_json("genesis_module_role_mapping.json"),
            "event_bus": self._load_json("event_bus.json"),
            "mutation_registry": self._load_json("strategy_mutation_registry.json") if os.path.exists(self.base_path / "strategy_mutation_registry.json") else {},
            "mutation_logbook": self._load_json("mutation_logbook.json") if os.path.exists(self.base_path / "mutation_logbook.json") else {}
        }
        
        emit_telemetry("trading_intelligence_audit", "audit_initialized", {
            "critical_modules_count": len(self.critical_trading_modules),
            "core_files_loaded": len([f for f in self.core_files.values() if f])
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
    
    def _read_module_source(self, module_path: str) -> Optional[str]:
        """Read source code of a module"""
        try:
            file_path = self.base_path / module_path
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return None
        except Exception as e:
            logger.error(f"Error reading {module_path}: {e}")
            return None
    
    def _analyze_module_for_stubs(self, source_code: str, module_name: str) -> Dict:
        """Analyze module source code for stubs, mocks, or placeholder logic"""
        if not source_code:
            return {"status": "MISSING", "violations": ["Module source not found"]}
        
        violations = []
        patterns_to_check = {
            "stub_patterns": [
                r"pass\s*$",
                r"TODO",
                r"raise NotImplementedError",
                r"return None\s*$",
                r"placeholder",
                r"stub",
                r"mock"
            ],
            "mock_data_patterns": [
                r"mock",
                r"simulate",
                r"test_data",
                r"dummy",
                r"sample_data",
                r"fake_"
            ],
            "fallback_patterns": [
                r"if.*fallback",
                r"default\s*=",
                r"except.*pass"
            ]
        }
        
        for category, patterns in patterns_to_check.items():
            for pattern in patterns:
                matches = re.findall(pattern, source_code, re.IGNORECASE | re.MULTILINE)
                if matches:
                    violations.append(f"{category}: {len(matches)} occurrences of '{pattern}'")
        
        # Check for actual trading logic indicators
        trading_logic_indicators = [
            r"mt5\.",
            r"MetaTrader5",
            r"order",
            r"trade",
            r"position",
            r"price",
            r"signal",
            r"strategy",
            r"risk",
            r"execute"
        ]
        
        trading_logic_count = 0
        for indicator in trading_logic_indicators:
            matches = re.findall(indicator, source_code, re.IGNORECASE)
            trading_logic_count += len(matches)
        
        # Analyze AST for function completeness
        try:
            tree = ast.parse(source_code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            empty_functions = 0
            for func in functions:
                # Check if function body only contains pass, docstring, or simple return
                body_statements = [stmt for stmt in func.body if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Constant)]
                if len(body_statements) <= 1:
                    empty_functions += 1
            
            if empty_functions > 0:
                violations.append(f"Empty/stub functions: {empty_functions}/{len(functions)}")
                
        except SyntaxError as e:
            violations.append(f"Syntax error in module: {e}")
        
        status = "COMPLETE" if not violations and trading_logic_count > 5 else "INCOMPLETE" if violations else "MINIMAL"
        
        return {
            "status": status,
            "violations": violations,
            "trading_logic_indicators": trading_logic_count,
            "source_lines": len(source_code.split('\n'))
        }
    
    def audit_critical_modules(self) -> Dict:
        """
        🔁 1. Audit all critical trading modules for completeness
        """
        logger.info("🔁 Phase 1: Critical Trading Modules Analysis")
        
        module_analysis = {}
        
        for module_name, possible_paths in self.critical_trading_modules.items():
            analysis = {
                "module_name": module_name,
                "found_at": None,
                "analysis_result": None,
                "eventbus_integration": False,
                "telemetry_integration": False,
                "mt5_integration": False
            }
            
            # Find the module
            for path in possible_paths:
                source_code = self._read_module_source(path)
                if source_code:
                    analysis["found_at"] = path
                    analysis["analysis_result"] = self._analyze_module_for_stubs(source_code, module_name)
                    
                    # Check for EventBus integration
                    if "emit_event" in source_code or "get_event_bus" in source_code:
                        analysis["eventbus_integration"] = True
                    
                    # Check for telemetry integration
                    if "emit_telemetry" in source_code or "telemetry" in source_code.lower():
                        analysis["telemetry_integration"] = True
                    
                    # Check for MT5 integration
                    if "mt5" in source_code.lower() or "MetaTrader5" in source_code:
                        analysis["mt5_integration"] = True
                    
                    break
            
            if not analysis["found_at"]:
                analysis["analysis_result"] = {"status": "MISSING", "violations": ["Module not found in expected locations"]}
            
            module_analysis[module_name] = analysis
        
        emit_telemetry("trading_intelligence_audit", "critical_modules_analyzed", {
            "modules_found": len([m for m in module_analysis.values() if m["found_at"]]),
            "modules_missing": len([m for m in module_analysis.values() if not m["found_at"]])
        })
        
        return module_analysis
    
    def validate_trading_flow(self) -> Dict:
        """
        🔁 2. Validate execution flow: macro → confluence → decision → risk → audit
        """
        logger.info("🔁 Phase 2: Trading Flow Validation")
        
        # Define expected trading flow
        trading_flow_steps = [
            {
                "step": "macro_analysis",
                "modules": ["macro_sync_engine"],
                "signals_produced": ["macro_analysis_complete", "economic_event_detected", "market_sentiment"],
                "signals_consumed": ["market_data_updated", "news_feed"]
            },
            {
                "step": "confluence_analysis", 
                "modules": ["pattern_classifier_engine", "signal_engine"],
                "signals_produced": ["confluence_score", "pattern_detected", "signal_generated"],
                "signals_consumed": ["macro_analysis_complete", "price_data", "indicator_data"]
            },
            {
                "step": "decision_logic",
                "modules": ["strategy_mutation_engine", "decision_engine"],
                "signals_produced": ["trade_decision", "entry_signal", "strategy_recommendation"],
                "signals_consumed": ["confluence_score", "pattern_detected", "risk_assessment"]
            },
            {
                "step": "risk_assessment",
                "modules": ["risk_guard", "risk_engine"],
                "signals_produced": ["risk_assessment", "position_sizing", "risk_approved"],
                "signals_consumed": ["trade_decision", "account_balance", "open_positions"]
            },
            {
                "step": "execution",
                "modules": ["execution_engine"],
                "signals_produced": ["order_placed", "execution_complete", "trade_opened"],
                "signals_consumed": ["risk_approved", "entry_signal", "position_sizing"]
            },
            {
                "step": "audit_monitoring",
                "modules": ["kill_switch_audit", "trade_journal_logger"],
                "signals_produced": ["trade_logged", "performance_metrics", "audit_complete"],
                "signals_consumed": ["trade_opened", "execution_complete", "position_update"]
            }
        ]
        
        flow_validation = {}
        role_mapping = self.core_files.get("role_mapping", {})
        
        for step_info in trading_flow_steps:
            step_name = step_info["step"]
            step_validation = {
                "modules_available": [],
                "modules_missing": [],
                "signal_producers": [],
                "signal_consumers": [],
                "flow_integrity": "UNKNOWN"
            }
            
            # Check if required modules exist and have signal capabilities
            for module_name in step_info["modules"]:
                if module_name in self.audit_report["critical_modules_analysis"]:
                    module_info = self.audit_report["critical_modules_analysis"][module_name]
                    if module_info["found_at"]:
                        step_validation["modules_available"].append(module_name)
                        
                        # Check signal production/consumption in role mapping
                        if "module_analysis" in role_mapping and module_name in role_mapping["module_analysis"]:
                            module_data = role_mapping["module_analysis"][module_name]
                            eventbus_routes = module_data.get("eventbus_routes", {})
                            
                            produced_signals = eventbus_routes.get("emitted", [])
                            consumed_signals = eventbus_routes.get("consumed", [])
                            
                            step_validation["signal_producers"].extend(produced_signals)
                            step_validation["signal_consumers"].extend(consumed_signals)
                    else:
                        step_validation["modules_missing"].append(module_name)
                else:
                    step_validation["modules_missing"].append(module_name)
            
            # Determine flow integrity
            if step_validation["modules_missing"]:
                step_validation["flow_integrity"] = "BROKEN"
            elif step_validation["signal_producers"] and step_validation["signal_consumers"]:
                step_validation["flow_integrity"] = "CONNECTED"
            elif step_validation["modules_available"]:
                step_validation["flow_integrity"] = "PARTIALLY_CONNECTED"
            else:
                step_validation["flow_integrity"] = "MISSING"
            
            flow_validation[step_name] = step_validation
        
        emit_telemetry("trading_intelligence_audit", "trading_flow_validated", flow_validation)
        
        return flow_validation
    
    def trace_signal_routes(self) -> Dict:
        """
        📡 3. Trace EventBus signal routes between trading modules
        """
        logger.info("📡 Phase 3: Signal Route Tracing")
        
        signal_trace = {
            "total_routes_traced": 0,
            "connected_routes": 0,
            "broken_routes": 0,
            "route_map": {},
            "orphan_signals": []
        }
        
        role_mapping = self.core_files.get("role_mapping", {})
        
        if "module_analysis" in role_mapping:
            # Get all trading-related modules
            trading_modules = {}
            for module_name in self.critical_trading_modules.keys():
                if module_name in role_mapping["module_analysis"]:
                    trading_modules[module_name] = role_mapping["module_analysis"][module_name]
            
            # Trace signals between trading modules
            for producer_module, producer_data in trading_modules.items():
                producer_signals = producer_data.get("eventbus_routes", {}).get("emitted", [])
                
                for signal in producer_signals:
                    signal_trace["total_routes_traced"] += 1
                    
                    # Find consumers for this signal
                    consumers = []
                    for consumer_module, consumer_data in trading_modules.items():
                        if consumer_module != producer_module:
                            consumer_signals = consumer_data.get("eventbus_routes", {}).get("consumed", [])
                            if signal in consumer_signals:
                                consumers.append(consumer_module)
                    
                    route_key = f"{producer_module}->{signal}"
                    if consumers:
                        signal_trace["connected_routes"] += 1
                        signal_trace["route_map"][route_key] = {
                            "producer": producer_module,
                            "signal": signal,
                            "consumers": consumers,
                            "status": "CONNECTED"
                        }
                    else:
                        signal_trace["broken_routes"] += 1
                        signal_trace["orphan_signals"].append({
                            "producer": producer_module,
                            "signal": signal,
                            "issue": "NO_TRADING_MODULE_CONSUMERS"
                        })
        
        emit_telemetry("trading_intelligence_audit", "signal_routes_traced", signal_trace)
        
        return signal_trace
    
    def simulate_execution_paths(self) -> Dict:
        """
        🧪 4. Simulate trade execution paths (intraday + swing scenarios)
        """
        logger.info("🧪 Phase 4: Execution Path Simulation")
        
        simulation_results = {
            "intraday_scenario": {},
            "swing_scenario": {},
            "execution_completeness": 0.0
        }
        
        # Define test scenarios
        scenarios = {
            "intraday_scenario": {
                "timeframe": "M15",
                "trade_type": "SCALP",
                "expected_signals": ["macro_analysis_complete", "confluence_score", "entry_signal", "risk_approved", "order_placed"],
                "expected_modules": ["macro_sync_engine", "pattern_classifier_engine", "risk_guard", "execution_engine"]
            },
            "swing_scenario": {
                "timeframe": "H4",
                "trade_type": "SWING",
                "expected_signals": ["pattern_detected", "strategy_recommendation", "position_sizing", "trade_opened"],
                "expected_modules": ["strategy_mutation_engine", "risk_guard", "execution_engine", "trade_journal_logger"]
            }
        }
        
        for scenario_name, scenario_config in scenarios.items():
            scenario_result = {
                "modules_available": 0,
                "modules_total": len(scenario_config["expected_modules"]),
                "signals_traceable": 0,
                "signals_total": len(scenario_config["expected_signals"]),
                "execution_path_complete": False,
                "missing_components": []
            }
            
            # Check module availability
            for module_name in scenario_config["expected_modules"]:
                if module_name in self.audit_report["critical_modules_analysis"]:
                    module_info = self.audit_report["critical_modules_analysis"][module_name]
                    if module_info["found_at"] and module_info["analysis_result"]["status"] in ["COMPLETE", "MINIMAL"]:
                        scenario_result["modules_available"] += 1
                    else:
                        scenario_result["missing_components"].append(f"Module: {module_name}")
                else:
                    scenario_result["missing_components"].append(f"Module: {module_name}")
            
            # Check signal traceability
            signal_trace = self.audit_report.get("signal_route_trace", {})
            route_map = signal_trace.get("route_map", {})
            
            for signal in scenario_config["expected_signals"]:
                # Check if signal exists in any route
                signal_found = False
                for route_key, route_info in route_map.items():
                    if route_info["signal"] == signal:
                        signal_found = True
                        break
                
                if signal_found:
                    scenario_result["signals_traceable"] += 1
                else:
                    scenario_result["missing_components"].append(f"Signal: {signal}")
            
            # Determine execution path completeness
            module_completeness = scenario_result["modules_available"] / scenario_result["modules_total"]
            signal_completeness = scenario_result["signals_traceable"] / scenario_result["signals_total"]
            overall_completeness = (module_completeness + signal_completeness) / 2
            
            scenario_result["execution_path_complete"] = overall_completeness >= 0.8
            scenario_result["completeness_score"] = overall_completeness * 100
            
            simulation_results[scenario_name] = scenario_result
        
        # Calculate overall execution completeness
        avg_completeness = (
            simulation_results["intraday_scenario"]["completeness_score"] +
            simulation_results["swing_scenario"]["completeness_score"]
        ) / 2
        
        simulation_results["execution_completeness"] = avg_completeness
        
        emit_telemetry("trading_intelligence_audit", "execution_paths_simulated", simulation_results)
        
        return simulation_results
    
    def generate_intelligence_score(self) -> float:
        """
        Calculate overall trading intelligence readiness score (0-100%)
        """
        logger.info("📊 Calculating Trading Intelligence Score")
        
        scoring_factors = {
            "module_completeness": 0.25,
            "flow_integrity": 0.20,
            "signal_connectivity": 0.20,
            "execution_readiness": 0.20,
            "architecture_compliance": 0.15
        }
        
        scores = {}
        
        # 1. Module Completeness Score
        module_analysis = self.audit_report["critical_modules_analysis"]
        complete_modules = len([m for m in module_analysis.values() 
                              if m["analysis_result"] and m["analysis_result"]["status"] == "COMPLETE"])
        total_modules = len(module_analysis)
        scores["module_completeness"] = (complete_modules / total_modules) * 100 if total_modules > 0 else 0
        
        # 2. Flow Integrity Score
        flow_validation = self.audit_report["trading_flow_validation"]
        connected_flows = len([f for f in flow_validation.values() if f["flow_integrity"] == "CONNECTED"])
        total_flows = len(flow_validation)
        scores["flow_integrity"] = (connected_flows / total_flows) * 100 if total_flows > 0 else 0
        
        # 3. Signal Connectivity Score
        signal_trace = self.audit_report["signal_route_trace"]
        total_routes = signal_trace.get("total_routes_traced", 0)
        connected_routes = signal_trace.get("connected_routes", 0)
        scores["signal_connectivity"] = (connected_routes / total_routes) * 100 if total_routes > 0 else 0
        
        # 4. Execution Readiness Score
        execution_tests = self.audit_report["execution_path_tests"]
        scores["execution_readiness"] = execution_tests.get("execution_completeness", 0)
        
        # 5. Architecture Compliance Score (EventBus + Telemetry + MT5 integration)
        compliance_score = 0
        compliance_factors = 0
        for module_info in module_analysis.values():
            if module_info["found_at"]:
                compliance_factors += 3  # EventBus, Telemetry, MT5
                if module_info["eventbus_integration"]:
                    compliance_score += 1
                if module_info["telemetry_integration"]:
                    compliance_score += 1
                if module_info["mt5_integration"]:
                    compliance_score += 1
        
        scores["architecture_compliance"] = (compliance_score / compliance_factors) * 100 if compliance_factors > 0 else 0
        
        # Calculate weighted total score
        total_score = sum(scores[factor] * weight for factor, weight in scoring_factors.items())
        
        self.audit_report["scoring_breakdown"] = scores
        
        return total_score
    
    def generate_recommendations(self) -> List[Dict]:
        """
        Generate actionable recommendations based on audit findings
        """
        logger.info("🔧 Generating Recommendations")
        
        recommendations = []
        
        # Analyze module issues
        module_analysis = self.audit_report["critical_modules_analysis"]
        for module_name, module_info in module_analysis.items():
            if not module_info["found_at"]:
                recommendations.append({
                    "priority": "CRITICAL",
                    "category": "MISSING_MODULE",
                    "module": module_name,
                    "action": f"Implement missing {module_name} module",
                    "details": f"Core trading module {module_name} not found in expected locations"
                })
            elif module_info["analysis_result"]["status"] in ["INCOMPLETE", "MINIMAL"]:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "MODULE_COMPLETION",
                    "module": module_name,
                    "action": f"Complete {module_name} implementation",
                    "details": f"Module contains stubs or insufficient trading logic: {module_info['analysis_result']['violations']}"
                })
            
            # Architecture compliance recommendations
            if module_info["found_at"] and not module_info["eventbus_integration"]:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "EVENTBUS_INTEGRATION",
                    "module": module_name,
                    "action": f"Add EventBus integration to {module_name}",
                    "details": "Module not connected to EventBus architecture"
                })
        
        # Signal routing recommendations
        signal_trace = self.audit_report["signal_route_trace"]
        for orphan in signal_trace.get("orphan_signals", []):
            recommendations.append({
                "priority": "MEDIUM",
                "category": "SIGNAL_ROUTING",
                "module": orphan["producer"],
                "action": f"Add consumer for signal '{orphan['signal']}'",
                "details": "Signal has no consumers in trading module ecosystem"
            })
        
        # Flow integrity recommendations
        flow_validation = self.audit_report["trading_flow_validation"]
        for step_name, step_info in flow_validation.items():
            if step_info["flow_integrity"] == "BROKEN":
                recommendations.append({
                    "priority": "CRITICAL",
                    "category": "FLOW_REPAIR",
                    "module": step_name,
                    "action": f"Repair broken trading flow step: {step_name}",
                    "details": f"Missing modules: {step_info['modules_missing']}"
                })
        
        return recommendations
    
    def run_trading_intelligence_audit(self) -> str:
        """
        Execute complete Phase 7.95 Trading Intelligence Audit
        """
        logger.info("🚀 Starting Phase 7.95 Trading Intelligence Audit")
        
        start_time = time.time()
        
        try:
            # Phase 1: Critical Modules Analysis
            self.audit_report["critical_modules_analysis"] = self.audit_critical_modules()
            
            # Phase 2: Trading Flow Validation
            self.audit_report["trading_flow_validation"] = self.validate_trading_flow()
            
            # Phase 3: Signal Route Tracing
            self.audit_report["signal_route_trace"] = self.trace_signal_routes()
            
            # Phase 4: Execution Path Simulation
            self.audit_report["execution_path_tests"] = self.simulate_execution_paths()
            
            # Phase 5: Generate Intelligence Score
            self.audit_report["trading_intelligence_score"] = self.generate_intelligence_score()
            
            # Phase 6: Generate Recommendations
            self.audit_report["recommendations"] = self.generate_recommendations()
            
            # Add execution metadata
            execution_time = time.time() - start_time
            self.audit_report["metadata"]["execution_time_seconds"] = execution_time
            self.audit_report["metadata"]["completion_status"] = "SUCCESS"
            
            # Save audit report
            report_filename = f"PHASE_7_TRADING_INTELLIGENCE_AUDIT_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = self.base_path / report_filename
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.audit_report, f, indent=2, ensure_ascii=False)
            
            # Create markdown summary
            self._create_markdown_report(report_filename.replace('.json', '.md'))
            
            emit_telemetry("trading_intelligence_audit", "audit_completed", {
                "execution_time": execution_time,
                "intelligence_score": self.audit_report["trading_intelligence_score"],
                "report_file": report_filename
            })
            
            logger.info(f"✅ Trading Intelligence Audit Complete - Report: {report_filename}")
            return report_filename
            
        except Exception as e:
            logger.error(f"❌ Audit failed: {e}")
            self.audit_report["metadata"]["completion_status"] = "FAILED"
            self.audit_report["metadata"]["error"] = str(e)
            
            # Save error report
            error_report_filename = f"TRADING_INTELLIGENCE_AUDIT_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(self.base_path / error_report_filename, 'w', encoding='utf-8') as f:
                json.dump(self.audit_report, f, indent=2, ensure_ascii=False)
            
            return error_report_filename
    
    def _create_markdown_report(self, filename: str):
        """Create a human-readable markdown summary"""
        intelligence_score = self.audit_report.get("trading_intelligence_score", 0)
        
        markdown_content = f"""# 🎯 GENESIS PHASE 7.95 TRADING INTELLIGENCE AUDIT REPORT

**Generated:** {self.audit_report['metadata']['timestamp']}  
**Architect Mode:** {self.audit_report['metadata']['architect_mode']}  
**Compliance Level:** {self.audit_report['metadata']['compliance_level']}

## 🧠 EXECUTIVE SUMMARY

| Metric | Value | Status |
|--------|-------|--------|
| **Trading Intelligence Score** | {intelligence_score:.1f}% | {'✅ PASS' if intelligence_score >= 80 else '⚠️ REVIEW' if intelligence_score >= 60 else '❌ FAIL'} |
| **Critical Modules Found** | {len([m for m in self.audit_report['critical_modules_analysis'].values() if m['found_at']])} / {len(self.audit_report['critical_modules_analysis'])} | {'✅' if all(m['found_at'] for m in self.audit_report['critical_modules_analysis'].values()) else '❌'} |
| **Trading Flow Integrity** | {len([f for f in self.audit_report['trading_flow_validation'].values() if f['flow_integrity'] == 'CONNECTED'])}/{len(self.audit_report['trading_flow_validation'])} Connected | {'✅' if all(f['flow_integrity'] == 'CONNECTED' for f in self.audit_report['trading_flow_validation'].values()) else '⚠️'} |
| **Signal Route Coverage** | {self.audit_report['signal_route_trace'].get('connected_routes', 0)}/{self.audit_report['signal_route_trace'].get('total_routes_traced', 0)} Connected | {'✅' if self.audit_report['signal_route_trace'].get('connected_routes', 0) > 0 else '❌'} |
| **Execution Path Readiness** | {self.audit_report['execution_path_tests'].get('execution_completeness', 0):.1f}% | {'✅' if self.audit_report['execution_path_tests'].get('execution_completeness', 0) >= 80 else '⚠️'} |

## 🔍 CRITICAL MODULES ANALYSIS

"""
        
        for module_name, module_info in self.audit_report['critical_modules_analysis'].items():
            status_icon = "✅" if module_info['found_at'] and module_info['analysis_result']['status'] == 'COMPLETE' else "⚠️" if module_info['found_at'] else "❌"
            markdown_content += f"- **{module_name}** {status_icon}\n"
            if module_info['found_at']:
                markdown_content += f"  - Location: `{module_info['found_at']}`\n"
                markdown_content += f"  - Status: {module_info['analysis_result']['status']}\n"
                if module_info['analysis_result']['violations']:
                    markdown_content += f"  - Issues: {', '.join(module_info['analysis_result']['violations'][:3])}\n"
            else:
                markdown_content += f"  - Status: MISSING\n"
        
        # Add trading flow summary
        markdown_content += f"""

## 🔄 TRADING FLOW VALIDATION

"""
        for step_name, step_info in self.audit_report['trading_flow_validation'].items():
            flow_icon = "✅" if step_info['flow_integrity'] == 'CONNECTED' else "⚠️" if step_info['flow_integrity'] == 'PARTIALLY_CONNECTED' else "❌"
            markdown_content += f"- **{step_name.replace('_', ' ').title()}** {flow_icon} ({step_info['flow_integrity']})\n"
        
        # Add top recommendations
        markdown_content += f"""

## 🔧 TOP RECOMMENDATIONS ({len(self.audit_report['recommendations'])})

"""
        for i, rec in enumerate(self.audit_report['recommendations'][:10]):  # Show first 10
            priority_icon = "🚨" if rec['priority'] == 'CRITICAL' else "⚠️" if rec['priority'] == 'HIGH' else "📌"
            markdown_content += f"{i+1}. {priority_icon} **{rec['category']}**: {rec['action']}\n"
        
        if len(self.audit_report['recommendations']) > 10:
            markdown_content += f"\n*...and {len(self.audit_report['recommendations']) - 10} more recommendations*\n"
        
        # Add scoring breakdown
        if "scoring_breakdown" in self.audit_report:
            markdown_content += f"""

## 📊 DETAILED SCORING BREAKDOWN

| Component | Score | Weight |
|-----------|-------|--------|
| **Module Completeness** | {self.audit_report['scoring_breakdown']['module_completeness']:.1f}% | 25% |
| **Flow Integrity** | {self.audit_report['scoring_breakdown']['flow_integrity']:.1f}% | 20% |
| **Signal Connectivity** | {self.audit_report['scoring_breakdown']['signal_connectivity']:.1f}% | 20% |
| **Execution Readiness** | {self.audit_report['scoring_breakdown']['execution_readiness']:.1f}% | 20% |
| **Architecture Compliance** | {self.audit_report['scoring_breakdown']['architecture_compliance']:.1f}% | 15% |

"""
        
        markdown_content += f"""

---
**Report generated by GENESIS Architect Mode v7.0.0 Trading Intelligence Audit Engine**  
**Execution Time:** {self.audit_report['metadata'].get('execution_time_seconds', 0):.2f} seconds
"""
        
        # Save markdown report
        with open(self.base_path / filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)


def main():
    """Main execution function"""
    print("🎯 GENESIS PHASE 7.95 TRADING INTELLIGENCE AUDIT ENGINE")
    print("=" * 60)
    
    auditor = TradingIntelligenceAuditEngine()
    report_filename = auditor.run_trading_intelligence_audit()
    
    print(f"\n✅ Trading Intelligence Audit Complete!")
    print(f"📄 Report saved as: {report_filename}")
    print(f"📄 Markdown summary: {report_filename.replace('.json', '.md')}")
    
    return report_filename


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Critical error during audit: {e}")
        sys.exit(1)
