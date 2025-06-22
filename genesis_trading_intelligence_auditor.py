#!/usr/bin/env python3
"""
GENESIS TRADING INTELLIGENCE AUDIT ANALYZER
============================================

Comprehensive audit of GENESIS algorithmic trading intelligence layer.
Validates operational readiness, execution paths, and live trading capabilities.

AUDIT SCOPE:
- Trading Logic Integrity & Completeness
- Execution Path Validation
- Strategy Engine Assessment  
- Live Environment Readiness
- Risk Management Systems
- Dashboard Integration Loops

Author: Genesis Trading Intelligence Audit System
Version: 1.0.0
Date: 2025-01-22
"""

import os
import json
import ast
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradingComponent:
    """Trading system component metadata"""
    name: str
    path: str
    type: str  # signal_generator, execution_engine, risk_manager, etc.
    complexity_score: float
    completeness_score: float
    live_ready: bool
    dependencies: List[str]
    issues: List[str]

@dataclass
class ExecutionPath:
    """Trading execution path analysis"""
    name: str
    start_trigger: str
    end_action: str
    steps: List[str]
    validation_points: List[str]
    completeness: float
    live_ready: bool

@dataclass
class TradingIntelligenceAudit:
    """Complete trading intelligence audit results"""
    timestamp: str
    overall_completeness: float
    live_readiness_score: float
    
    # Core Components
    signal_generators: List[TradingComponent]
    execution_engines: List[TradingComponent]
    risk_managers: List[TradingComponent]
    strategy_engines: List[TradingComponent]
    
    # Execution Paths
    execution_paths: List[ExecutionPath]
    
    # Integration Assessment
    mt5_integration: Dict[str, Any]
    dashboard_loops: Dict[str, Any]
    event_bus_trading: Dict[str, Any]
    
    # Critical Issues
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Metrics
    total_trading_files: int
    analyzed_files: int
    execution_completeness: float
    risk_coverage: float

class TradingIntelligenceAuditor:
    """Advanced trading intelligence audit analyzer"""
    
    def __init__(self, genesis_root: str):
        self.genesis_root = Path(genesis_root)
        self.trading_components = []
        self.execution_paths = []
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
        # Trading-specific patterns
        self.trading_patterns = {
            'signal_generation': [
                r'generate_signal', r'signal_strength', r'entry_signal',
                r'exit_signal', r'trend_analysis', r'momentum_calc'
            ],
            'execution_logic': [
                r'execute_trade', r'place_order', r'modify_order',
                r'close_position', r'partial_close', r'trade_execution'
            ],
            'risk_management': [
                r'stop_loss', r'take_profit', r'position_size',
                r'risk_calc', r'drawdown_check', r'exposure_limit'
            ],
            'strategy_engine': [
                r'strategy_mutation', r'parameter_optimization', r'adaptive_logic',
                r'performance_feedback', r'strategy_evolution'
            ],
            'mt5_integration': [
                r'MT5.*connect', r'symbol_info', r'market_data',
                r'tick_data', r'account_info', r'positions_get'
            ],
            'kill_switch': [
                r'emergency_stop', r'kill_switch', r'force_close',
                r'panic_mode', r'circuit_breaker', r'emergency_exit'
            ]
        }
        
        # Critical trading files to analyze
        self.critical_files = [
            'execution_engine.py', 'risk_guard.py', 'kill_switch_audit.py',
            'signal_generator.py', 'strategy_engine.py', 'mt5_connector.py',
            'position_manager.py', 'order_manager.py', 'portfolio_manager.py'
        ]
        
    def analyze_trading_intelligence(self) -> TradingIntelligenceAudit:
        """Execute comprehensive trading intelligence audit"""
        logger.info("🚀 Starting GENESIS Trading Intelligence Audit...")
        
        # Scan for trading files
        trading_files = self._find_trading_files()
        logger.info(f"📁 Found {len(trading_files)} trading-related files")
        
        # Analyze core components
        self._analyze_signal_generators(trading_files)
        self._analyze_execution_engines(trading_files)
        self._analyze_risk_managers(trading_files)
        self._analyze_strategy_engines(trading_files)
        
        # Analyze execution paths
        self._analyze_execution_paths(trading_files)
        
        # Check integrations
        mt5_integration = self._assess_mt5_integration(trading_files)
        dashboard_loops = self._assess_dashboard_integration(trading_files)
        event_bus_trading = self._assess_event_bus_trading(trading_files)
        
        # Calculate overall scores
        overall_completeness = self._calculate_overall_completeness()
        live_readiness = self._calculate_live_readiness()
        execution_completeness = self._calculate_execution_completeness()
        risk_coverage = self._calculate_risk_coverage()
        
        # Create audit result
        audit = TradingIntelligenceAudit(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_completeness=overall_completeness,
            live_readiness_score=live_readiness,
            signal_generators=self._get_components_by_type('signal_generator'),
            execution_engines=self._get_components_by_type('execution_engine'),
            risk_managers=self._get_components_by_type('risk_manager'),
            strategy_engines=self._get_components_by_type('strategy_engine'),
            execution_paths=self.execution_paths,
            mt5_integration=mt5_integration,
            dashboard_loops=dashboard_loops,
            event_bus_trading=event_bus_trading,
            critical_issues=self.critical_issues,
            warnings=self.warnings,
            recommendations=self.recommendations,
            total_trading_files=len(trading_files),
            analyzed_files=len([c for c in self.trading_components if c.completeness_score > 0]),
            execution_completeness=execution_completeness,
            risk_coverage=risk_coverage        )
        
        logger.info(f"✅ Trading Intelligence Audit Complete!")
        logger.info(f"📊 Overall Completeness: {overall_completeness:.1f}%")
        logger.info(f"🔥 Live Readiness: {live_readiness:.1f}%")
        
        return audit
    
    def _find_trading_files(self) -> List[Path]:
        """Find critical trading-related Python files (optimized)"""
        trading_files = []
        
        # Prioritize critical files first
        for root, dirs, files in os.walk(self.genesis_root):
            root_path = Path(root)
            
            for file in files:
                if file in self.critical_files and file.endswith('.py'):
                    trading_files.append(root_path / file)
        
        # Then look for files in specific trading directories (limited depth)
        trading_dirs = [
            'trading', 'execution', 'strategy', 'risk', 'signals',
            'orders', 'portfolio', 'mt5', 'broker', 'engine'
        ]
        
        for trading_dir in trading_dirs:
            for root, dirs, files in os.walk(self.genesis_root):
                root_path = Path(root)
                
                # Only check first 2 levels to avoid deep recursion
                depth = len(root_path.parts) - len(self.genesis_root.parts)
                if depth > 2:
                    continue
                
                if trading_dir in str(root_path).lower():
                    for file in files:
                        if file.endswith('.py'):
                            file_path = root_path / file
                            if file_path not in trading_files:
                                trading_files.append(file_path)
                
                # Limit to 500 files for performance
                if len(trading_files) > 500:
                    break
            
            if len(trading_files) > 500:
                break
        
        return trading_files[:500]  # Hard limit for performance
    
    def _contains_trading_logic(self, file_path: Path) -> bool:
        """Check if file contains trading-related logic"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
            # Check for trading keywords
            trading_keywords = [
                'trade', 'order', 'position', 'signal', 'strategy',
                'mt5', 'metatrader', 'forex', 'price', 'market',
                'buy', 'sell', 'stop_loss', 'take_profit', 'execution'
            ]
            
            return any(keyword in content for keyword in trading_keywords)
            
        except Exception:
            return False
    
    def _analyze_signal_generators(self, trading_files: List[Path]):
        """Analyze signal generation components"""
        for file_path in trading_files:
            if self._is_signal_generator(file_path):
                component = self._analyze_trading_component(file_path, 'signal_generator')
                self.trading_components.append(component)
    
    def _analyze_execution_engines(self, trading_files: List[Path]):
        """Analyze execution engine components"""
        for file_path in trading_files:
            if self._is_execution_engine(file_path):
                component = self._analyze_trading_component(file_path, 'execution_engine')
                self.trading_components.append(component)
    
    def _analyze_risk_managers(self, trading_files: List[Path]):
        """Analyze risk management components"""
        for file_path in trading_files:
            if self._is_risk_manager(file_path):
                component = self._analyze_trading_component(file_path, 'risk_manager')
                self.trading_components.append(component)
    
    def _analyze_strategy_engines(self, trading_files: List[Path]):
        """Analyze strategy engine components"""
        for file_path in trading_files:
            if self._is_strategy_engine(file_path):
                component = self._analyze_trading_component(file_path, 'strategy_engine')
                self.trading_components.append(component)
    
    def _is_signal_generator(self, file_path: Path) -> bool:
        """Check if file is a signal generator"""
        filename = file_path.name.lower()
        signal_indicators = ['signal', 'indicator', 'analysis', 'technical']
        return any(indicator in filename for indicator in signal_indicators)
    
    def _is_execution_engine(self, file_path: Path) -> bool:
        """Check if file is an execution engine"""
        filename = file_path.name.lower()
        execution_indicators = ['execution', 'engine', 'trade', 'order']
        return any(indicator in filename for indicator in execution_indicators)
    
    def _is_risk_manager(self, file_path: Path) -> bool:
        """Check if file is a risk manager"""
        filename = file_path.name.lower()
        risk_indicators = ['risk', 'guard', 'kill', 'stop', 'limit']
        return any(indicator in filename for indicator in risk_indicators)
    
    def _is_strategy_engine(self, file_path: Path) -> bool:
        """Check if file is a strategy engine"""
        filename = file_path.name.lower()
        strategy_indicators = ['strategy', 'adaptive', 'mutation', 'optimization']
        return any(indicator in filename for indicator in strategy_indicators)
    
    def _analyze_trading_component(self, file_path: Path, component_type: str) -> TradingComponent:
        """Analyze individual trading component"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for deep analysis
            try:
                tree = ast.parse(content)
                complexity_score = self._calculate_complexity(tree)
                completeness_score = self._calculate_completeness(tree, component_type)
                live_ready = self._assess_live_readiness(tree, content)
                dependencies = self._extract_dependencies(tree)
                issues = self._identify_component_issues(tree, content, component_type)
                
            except SyntaxError:
                complexity_score = 0.0
                completeness_score = 0.0
                live_ready = False
                dependencies = []
                issues = ["Syntax error in file"]
            
            return TradingComponent(
                name=file_path.name,
                path=str(file_path),
                type=component_type,
                complexity_score=complexity_score,
                completeness_score=completeness_score,
                live_ready=live_ready,
                dependencies=dependencies,
                issues=issues
            )
            
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return TradingComponent(
                name=file_path.name,
                path=str(file_path),
                type=component_type,
                complexity_score=0.0,
                completeness_score=0.0,
                live_ready=False,
                dependencies=[],
                issues=[f"Analysis error: {str(e)}"]
            )
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate code complexity score"""
        complexity_metrics = {
            'classes': 0,
            'functions': 0,
            'loops': 0,
            'conditions': 0,
            'lines': 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                complexity_metrics['classes'] += 1
            elif isinstance(node, ast.FunctionDef):
                complexity_metrics['functions'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                complexity_metrics['loops'] += 1
            elif isinstance(node, ast.If):
                complexity_metrics['conditions'] += 1
        
        # Normalize to 0-100 scale
        base_score = (
            complexity_metrics['classes'] * 10 +
            complexity_metrics['functions'] * 5 +
            complexity_metrics['loops'] * 3 +
            complexity_metrics['conditions'] * 2
        )
        
        return min(100.0, base_score)
    
    def _calculate_completeness(self, tree: ast.AST, component_type: str) -> float:
        """Calculate component completeness based on expected patterns"""
        required_patterns = self.trading_patterns.get(component_type, [])
        if not required_patterns:
            return 50.0  # Default for unknown types
        
        # Convert AST to string for pattern matching
        content = ast.unparse(tree) if hasattr(ast, 'unparse') else ""
        
        found_patterns = 0
        for pattern in required_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_patterns += 1
        
        return (found_patterns / len(required_patterns)) * 100.0
    
    def _assess_live_readiness(self, tree: ast.AST, content: str) -> bool:
        """Assess if component is ready for live trading"""
        # Check for live trading indicators
        live_indicators = [
            r'real_time', r'live_data', r'mt5_connect',
            r'production', r'live_trading', r'real_account'
        ]
        
        # Check for demo/test indicators (negative for live readiness)
        demo_indicators = [
            r'demo', r'test', r'mock', r'simulation',
            r'fake', r'stub', r'placeholder'
        ]
        
        has_live = any(re.search(pattern, content, re.IGNORECASE) for pattern in live_indicators)
        has_demo = any(re.search(pattern, content, re.IGNORECASE) for pattern in demo_indicators)
        
        # Must have live indicators and minimal demo/test code
        return has_live and not has_demo
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract component dependencies"""
        dependencies = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        
        return list(set(dependencies))
    
    def _identify_component_issues(self, tree: ast.AST, content: str, component_type: str) -> List[str]:
        """Identify issues in trading component"""
        issues = []
        
        # Check for TODO/FIXME comments
        if re.search(r'(TODO|FIXME|HACK|XXX)', content, re.IGNORECASE):
            issues.append("Contains TODO/FIXME comments")
        
        # Check for hardcoded values
        if re.search(r'(password|key|secret)\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
            issues.append("Contains hardcoded credentials")
        
        # Check for error handling
        has_try_except = any(isinstance(node, ast.Try) for node in ast.walk(tree))
        if not has_try_except:
            issues.append("Missing error handling")
        
        # Component-specific checks
        if component_type == 'execution_engine':
            if not re.search(r'stop_loss|take_profit', content, re.IGNORECASE):
                issues.append("Missing SL/TP implementation")
        
        if component_type == 'risk_manager':
            if not re.search(r'position_size|risk_calc', content, re.IGNORECASE):
                issues.append("Missing position sizing logic")
        
        return issues
    
    def _analyze_execution_paths(self, trading_files: List[Path]):
        """Analyze trading execution paths"""
        # Define critical execution paths
        critical_paths = [
            {
                'name': 'Signal_to_Order_Execution',
                'start_trigger': 'signal_generated',
                'end_action': 'order_placed',
                'expected_steps': ['signal_validation', 'risk_check', 'position_sizing', 'order_creation']
            },
            {
                'name': 'Position_Management',
                'start_trigger': 'position_opened',
                'end_action': 'position_closed',
                'expected_steps': ['sl_tp_set', 'monitoring', 'modification', 'close_trigger']
            },
            {
                'name': 'Risk_Kill_Switch',
                'start_trigger': 'risk_breach',
                'end_action': 'emergency_close',
                'expected_steps': ['risk_detection', 'kill_switch_trigger', 'force_close_all']
            }
        ]
        
        for path_def in critical_paths:
            execution_path = self._trace_execution_path(trading_files, path_def)
            self.execution_paths.append(execution_path)
    
    def _trace_execution_path(self, trading_files: List[Path], path_def: Dict) -> ExecutionPath:
        """Trace a specific execution path through the codebase"""
        found_steps = []
        validation_points = []
        
        # Search for path components in files
        for file_path in trading_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for start trigger
                if re.search(path_def['start_trigger'], content, re.IGNORECASE):
                    found_steps.append(f"START: {path_def['start_trigger']} in {file_path.name}")
                
                # Check for expected steps
                for step in path_def['expected_steps']:
                    if re.search(step, content, re.IGNORECASE):
                        found_steps.append(f"STEP: {step} in {file_path.name}")
                        validation_points.append(step)
                
                # Check for end action
                if re.search(path_def['end_action'], content, re.IGNORECASE):
                    found_steps.append(f"END: {path_def['end_action']} in {file_path.name}")
                
            except Exception:
                continue
        
        # Calculate completeness
        expected_count = 2 + len(path_def['expected_steps'])  # start + steps + end
        completeness = (len(found_steps) / expected_count) * 100.0 if expected_count > 0 else 0.0
        
        # Assess live readiness
        live_ready = completeness > 75.0 and len(validation_points) >= len(path_def['expected_steps']) * 0.7
        
        return ExecutionPath(
            name=path_def['name'],
            start_trigger=path_def['start_trigger'],
            end_action=path_def['end_action'],
            steps=found_steps,
            validation_points=validation_points,
            completeness=completeness,
            live_ready=live_ready
        )
    
    def _assess_mt5_integration(self, trading_files: List[Path]) -> Dict[str, Any]:
        """Assess MT5 integration completeness"""
        mt5_features = {
            'connection': False,
            'account_info': False,
            'symbol_info': False,
            'market_data': False,
            'order_send': False,
            'positions_get': False,
            'history_get': False
        }
        
        for file_path in trading_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for MT5 features
                if re.search(r'mt5.*connect|MetaTrader5.*connect', content, re.IGNORECASE):
                    mt5_features['connection'] = True
                if re.search(r'account_info|account_get', content, re.IGNORECASE):
                    mt5_features['account_info'] = True
                if re.search(r'symbol_info|symbols_get', content, re.IGNORECASE):
                    mt5_features['symbol_info'] = True
                if re.search(r'copy_rates|copy_ticks', content, re.IGNORECASE):
                    mt5_features['market_data'] = True
                if re.search(r'order_send|order_place', content, re.IGNORECASE):
                    mt5_features['order_send'] = True
                if re.search(r'positions_get|positions_total', content, re.IGNORECASE):
                    mt5_features['positions_get'] = True
                if re.search(r'history_deals|history_orders', content, re.IGNORECASE):
                    mt5_features['history_get'] = True
                    
            except Exception:
                continue
        
        completeness = (sum(mt5_features.values()) / len(mt5_features)) * 100.0
        
        return {
            'features': mt5_features,
            'completeness': completeness,
            'live_ready': completeness > 85.0
        }
    
    def _assess_dashboard_integration(self, trading_files: List[Path]) -> Dict[str, Any]:
        """Assess dashboard integration loops"""
        dashboard_features = {
            'real_time_updates': False,
            'websocket_connection': False,
            'event_emission': False,
            'status_reporting': False,
            'performance_metrics': False
        }
        
        for file_path in trading_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if re.search(r'real_time|live_update', content, re.IGNORECASE):
                    dashboard_features['real_time_updates'] = True
                if re.search(r'websocket|socketio', content, re.IGNORECASE):
                    dashboard_features['websocket_connection'] = True
                if re.search(r'emit|broadcast|publish', content, re.IGNORECASE):
                    dashboard_features['event_emission'] = True
                if re.search(r'status|state|health', content, re.IGNORECASE):
                    dashboard_features['status_reporting'] = True
                if re.search(r'performance|metrics|analytics', content, re.IGNORECASE):
                    dashboard_features['performance_metrics'] = True
                    
            except Exception:
                continue
        
        completeness = (sum(dashboard_features.values()) / len(dashboard_features)) * 100.0
        
        return {
            'features': dashboard_features,
            'completeness': completeness,
            'live_ready': completeness > 70.0
        }
      def _assess_event_bus_trading(self, trading_files: List[Path]) -> Dict[str, Any]:
        """Assess EventBus integration for trading (optimized)"""
        event_patterns = {
            'signal_events': False,
            'execution_events': False,
            'risk_events': False,
            'position_events': False,
            'error_events': False
        }
        
        # Limit files to process for performance
        files_to_process = trading_files[:100] if len(trading_files) > 100 else trading_files
        
        for file_path in files_to_process:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use simpler string checks for better performance
                content_lower = content.lower()
                
                if 'signal' in content_lower and 'event' in content_lower:
                    event_patterns['signal_events'] = True
                if ('execution' in content_lower or 'trade' in content_lower) and 'event' in content_lower:
                    event_patterns['execution_events'] = True
                if ('risk' in content_lower or 'stop' in content_lower) and 'event' in content_lower:
                    event_patterns['risk_events'] = True
                if ('position' in content_lower or 'order' in content_lower) and 'event' in content_lower:
                    event_patterns['position_events'] = True
                if ('error' in content_lower or 'exception' in content_lower) and 'event' in content_lower:
                    event_patterns['error_events'] = True
                    
            except Exception:
                continue
        
        completeness = (sum(event_patterns.values()) / len(event_patterns)) * 100.0
        
        return {
            'patterns': event_patterns,
            'completeness': completeness,
            'live_ready': completeness > 60.0
        }
    
    def _calculate_overall_completeness(self) -> float:
        """Calculate overall trading intelligence completeness"""
        if not self.trading_components:
            return 0.0
        
        total_completeness = sum(c.completeness_score for c in self.trading_components)
        return total_completeness / len(self.trading_components)
    
    def _calculate_live_readiness(self) -> float:
        """Calculate live trading readiness score"""
        if not self.trading_components:
            return 0.0
        
        live_ready_count = sum(1 for c in self.trading_components if c.live_ready)
        return (live_ready_count / len(self.trading_components)) * 100.0
    
    def _calculate_execution_completeness(self) -> float:
        """Calculate execution path completeness"""
        if not self.execution_paths:
            return 0.0
        
        total_completeness = sum(path.completeness for path in self.execution_paths)
        return total_completeness / len(self.execution_paths)
    
    def _calculate_risk_coverage(self) -> float:
        """Calculate risk management coverage"""
        risk_managers = self._get_components_by_type('risk_manager')
        if not risk_managers:
            return 0.0
        
        total_completeness = sum(rm.completeness_score for rm in risk_managers)
        return total_completeness / len(risk_managers)
    
    def _get_components_by_type(self, component_type: str) -> List[TradingComponent]:
        """Get all components of specified type"""
        return [c for c in self.trading_components if c.type == component_type]

def generate_trading_intelligence_report(audit: TradingIntelligenceAudit, output_dir: str):
    """Generate comprehensive trading intelligence audit report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON Report
    json_path = os.path.join(output_dir, f"trading_intelligence_audit_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(asdict(audit), f, indent=2, default=str)
    
    # Markdown Report
    md_path = os.path.join(output_dir, f"TRADING_INTELLIGENCE_AUDIT_REPORT_{timestamp}.md")
    
    with open(md_path, 'w') as f:
        f.write(f"""# GENESIS TRADING INTELLIGENCE AUDIT REPORT
## Generated: {audit.timestamp}

---

## 🎯 EXECUTIVE SUMMARY

**Overall Trading Intelligence Completeness**: {audit.overall_completeness:.1f}%
**Live Trading Readiness Score**: {audit.live_readiness_score:.1f}%
**Execution Path Completeness**: {audit.execution_completeness:.1f}%
**Risk Management Coverage**: {audit.risk_coverage:.1f}%

### 📊 SYSTEM STATUS
- **Total Trading Files Analyzed**: {audit.total_trading_files}
- **Successfully Analyzed**: {audit.analyzed_files}
- **Analysis Coverage**: {(audit.analyzed_files/audit.total_trading_files*100) if audit.total_trading_files > 0 else 0:.1f}%

---

## 🔥 CRITICAL ASSESSMENT

### Trading Component Analysis
""")

        # Signal Generators
        f.write(f"""
#### 📡 Signal Generators ({len(audit.signal_generators)})
""")
        for sg in audit.signal_generators:
            f.write(f"""
- **{sg.name}**
  - Completeness: {sg.completeness_score:.1f}%
  - Live Ready: {'✅' if sg.live_ready else '❌'}
  - Issues: {len(sg.issues)}
""")

        # Execution Engines  
        f.write(f"""
#### ⚡ Execution Engines ({len(audit.execution_engines)})
""")
        for ee in audit.execution_engines:
            f.write(f"""
- **{ee.name}**
  - Completeness: {ee.completeness_score:.1f}%
  - Live Ready: {'✅' if ee.live_ready else '❌'}
  - Issues: {len(ee.issues)}
""")

        # Risk Managers
        f.write(f"""
#### 🛡️ Risk Managers ({len(audit.risk_managers)})
""")
        for rm in audit.risk_managers:
            f.write(f"""
- **{rm.name}**
  - Completeness: {rm.completeness_score:.1f}%
  - Live Ready: {'✅' if rm.live_ready else '❌'}
  - Issues: {len(rm.issues)}
""")

        # Strategy Engines
        f.write(f"""
#### 🧠 Strategy Engines ({len(audit.strategy_engines)})
""")
        for se in audit.strategy_engines:
            f.write(f"""
- **{se.name}**
  - Completeness: {se.completeness_score:.1f}%
  - Live Ready: {'✅' if se.live_ready else '❌'}
  - Issues: {len(se.issues)}
""")

        # Execution Paths
        f.write(f"""
---

## 🔄 EXECUTION PATH ANALYSIS

""")
        for path in audit.execution_paths:
            f.write(f"""
### {path.name}
- **Trigger**: {path.start_trigger} → **Action**: {path.end_action}
- **Completeness**: {path.completeness:.1f}%
- **Live Ready**: {'✅' if path.live_ready else '❌'}
- **Validation Points**: {len(path.validation_points)}
- **Steps Found**: {len(path.steps)}

""")

        # Integration Assessment
        f.write(f"""
---

## 🔗 INTEGRATION ASSESSMENT

### MT5 Integration
- **Completeness**: {audit.mt5_integration['completeness']:.1f}%
- **Live Ready**: {'✅' if audit.mt5_integration['live_ready'] else '❌'}
- **Features**: {sum(audit.mt5_integration['features'].values())}/{len(audit.mt5_integration['features'])}

### Dashboard Integration Loops
- **Completeness**: {audit.dashboard_loops['completeness']:.1f}%
- **Live Ready**: {'✅' if audit.dashboard_loops['live_ready'] else '❌'}
- **Features**: {sum(audit.dashboard_loops['features'].values())}/{len(audit.dashboard_loops['features'])}

### EventBus Trading Integration
- **Completeness**: {audit.event_bus_trading['completeness']:.1f}%
- **Live Ready**: {'✅' if audit.event_bus_trading['live_ready'] else '❌'}
- **Patterns**: {sum(audit.event_bus_trading['patterns'].values())}/{len(audit.event_bus_trading['patterns'])}

---

## ⚠️ CRITICAL ISSUES
""")

        for issue in audit.critical_issues:
            f.write(f"- 🚨 {issue}\n")

        f.write(f"""
## ⚡ WARNINGS
""")
        for warning in audit.warnings:
            f.write(f"- ⚠️ {warning}\n")

        f.write(f"""
## 💡 RECOMMENDATIONS
""")
        for rec in audit.recommendations:
            f.write(f"- 💡 {rec}\n")

        f.write(f"""
---

## 🎯 FINAL VERDICT

**TRADING INTELLIGENCE STATUS**: {'🔥 PRODUCTION READY' if audit.overall_completeness > 80 and audit.live_readiness_score > 75 else '🚧 NEEDS DEVELOPMENT' if audit.overall_completeness > 50 else '❌ INCOMPLETE'}

**LIVE TRADING READINESS**: {'✅ READY' if audit.live_readiness_score > 75 else '⚠️ PARTIAL' if audit.live_readiness_score > 50 else '❌ NOT READY'}

**NEXT ACTIONS**: {'Focus on integration fixes and testing' if audit.overall_completeness > 70 else 'Complete core trading logic development'}

---
*Generated by GENESIS Trading Intelligence Audit System v1.0.0*
""")

    logger.info(f"📄 Trading Intelligence Audit Report saved to: {md_path}")
    logger.info(f"📄 Trading Intelligence Data saved to: {json_path}")
    
    return md_path, json_path

def main():
    """Main audit execution"""
    # Configuration
    genesis_root = r"C:\Users\patra\Genesis FINAL TRY"
    
    print("🚀 GENESIS TRADING INTELLIGENCE AUDIT")
    print("=====================================")
    print(f"📁 Analyzing: {genesis_root}")
    print()
    
    # Initialize auditor
    auditor = TradingIntelligenceAuditor(genesis_root)
    
    # Execute comprehensive audit
    audit_result = auditor.analyze_trading_intelligence()
    
    # Generate reports
    md_report, json_report = generate_trading_intelligence_report(audit_result, genesis_root)
    
    print()
    print("📊 AUDIT COMPLETE!")
    print(f"🎯 Overall Completeness: {audit_result.overall_completeness:.1f}%")
    print(f"🔥 Live Readiness: {audit_result.live_readiness_score:.1f}%") 
    print(f"⚡ Execution Completeness: {audit_result.execution_completeness:.1f}%")
    print(f"🛡️ Risk Coverage: {audit_result.risk_coverage:.1f}%")
    print()
    print(f"📄 Report: {md_report}")
    print(f"📄 Data: {json_report}")

if __name__ == "__main__":
    main()
