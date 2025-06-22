#!/usr/bin/env python3
"""
GENESIS TRADING INTELLIGENCE AUDIT ANALYZER (OPTIMIZED)
======================================================

Fast and focused audit of GENESIS algorithmic trading intelligence layer.
Validates operational readiness and execution paths for live trading.

Author: Genesis Trading Intelligence Audit System
Version: 1.1.0 (Optimized)
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingIntelligenceAudit:
    """Trading intelligence audit results"""
    timestamp: str
    overall_completeness: float
    live_readiness_score: float
    critical_files_found: int
    critical_files_analyzed: int
    mt5_integration_score: float
    execution_paths_score: float
    risk_coverage_score: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

class OptimizedTradingAuditor:
    """Fast trading intelligence auditor focused on critical components"""
    
    def __init__(self, genesis_root: str):
        self.genesis_root = Path(genesis_root)
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
        # Critical trading files to find and analyze
        self.critical_files = {
            'execution_engine.py': 'Core execution logic',
            'risk_guard.py': 'Risk management',
            'kill_switch_audit.py': 'Emergency controls',
            'signal_generator.py': 'Trading signals',
            'strategy_engine.py': 'Strategy logic',
            'mt5_connector.py': 'MT5 integration',
            'position_manager.py': 'Position management',
            'order_manager.py': 'Order handling',
            'portfolio_manager.py': 'Portfolio logic'
        }
        
        # Trading intelligence patterns to detect
        self.intelligence_patterns = {
            'signal_generation': r'(generate_signal|signal_strength|entry_signal|exit_signal)',
            'execution_logic': r'(execute_trade|place_order|modify_order|close_position)',
            'risk_management': r'(stop_loss|take_profit|position_size|risk_calc)',
            'mt5_integration': r'(MT5|MetaTrader5|mt5_connect|copy_rates)',
            'live_trading': r'(real_time|live_data|production|live_trading)',
            'kill_switch': r'(emergency_stop|kill_switch|force_close|panic_mode)'
        }
    
    def run_comprehensive_audit(self) -> TradingIntelligenceAudit:
        """Execute fast comprehensive trading intelligence audit"""
        logger.info("🚀 Starting Optimized GENESIS Trading Intelligence Audit...")
        
        # Find critical trading files
        found_files = self._find_critical_files()
        logger.info(f"📁 Found {len(found_files)} critical trading files")
        
        # Analyze core trading components
        analysis_results = self._analyze_trading_components(found_files)
        
        # Assess integrations
        mt5_score = self._assess_mt5_integration(found_files)
        execution_score = self._assess_execution_paths(found_files)
        risk_score = self._assess_risk_coverage(found_files)
        
        # Calculate overall scores
        overall_completeness = self._calculate_overall_completeness(analysis_results)
        live_readiness = self._calculate_live_readiness(analysis_results)
        
        # Generate final assessment
        self._generate_final_assessment(overall_completeness, live_readiness, mt5_score)
        
        audit = TradingIntelligenceAudit(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_completeness=overall_completeness,
            live_readiness_score=live_readiness,
            critical_files_found=len(found_files),
            critical_files_analyzed=len([f for f in found_files if f['analyzed']]),
            mt5_integration_score=mt5_score,
            execution_paths_score=execution_score,
            risk_coverage_score=risk_score,
            critical_issues=self.critical_issues,
            warnings=self.warnings,
            recommendations=self.recommendations
        )
        
        logger.info(f"✅ Trading Intelligence Audit Complete!")
        logger.info(f"📊 Overall Completeness: {overall_completeness:.1f}%")
        logger.info(f"🔥 Live Readiness: {live_readiness:.1f}%")
        
        return audit
    
    def _find_critical_files(self) -> List[Dict]:
        """Find critical trading files in the codebase"""
        found_files = []
        
        logger.info("🔍 Searching for critical trading files...")
        
        for root, dirs, files in os.walk(self.genesis_root):
            for file in files:
                if file in self.critical_files and file.endswith('.py'):
                    file_path = Path(root) / file
                    found_files.append({
                        'name': file,
                        'path': str(file_path),
                        'description': self.critical_files[file],
                        'analyzed': False,
                        'completeness': 0.0,
                        'live_ready': False,
                        'issues': []
                    })
        
        return found_files
    
    def _analyze_trading_components(self, found_files: List[Dict]) -> List[Dict]:
        """Analyze trading components for completeness and readiness"""
        logger.info("🔬 Analyzing trading components...")
        
        for file_info in found_files:
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze content
                completeness = self._calculate_file_completeness(content, file_info['name'])
                live_ready = self._assess_file_live_readiness(content)
                issues = self._identify_file_issues(content, file_info['name'])
                
                # Update file info
                file_info['analyzed'] = True
                file_info['completeness'] = completeness
                file_info['live_ready'] = live_ready
                file_info['issues'] = issues
                
                logger.info(f"📄 {file_info['name']}: {completeness:.1f}% complete, Live Ready: {live_ready}")
                
            except Exception as e:
                file_info['issues'].append(f"Analysis error: {str(e)}")
                logger.warning(f"⚠️ Error analyzing {file_info['name']}: {e}")
        
        return found_files
    
    def _calculate_file_completeness(self, content: str, filename: str) -> float:
        """Calculate completeness score for a trading file"""
        # Define expected patterns based on file type
        expected_patterns = []
        
        if 'execution' in filename.lower():
            expected_patterns = ['execute_trade', 'place_order', 'modify_order', 'close_position']
        elif 'risk' in filename.lower() or 'guard' in filename.lower():
            expected_patterns = ['stop_loss', 'take_profit', 'position_size', 'risk_calc']
        elif 'signal' in filename.lower():
            expected_patterns = ['generate_signal', 'signal_strength', 'entry_signal']
        elif 'strategy' in filename.lower():
            expected_patterns = ['strategy_mutation', 'parameter_optimization', 'adaptive_logic']
        elif 'mt5' in filename.lower() or 'connector' in filename.lower():
            expected_patterns = ['MT5', 'connect', 'copy_rates', 'order_send']
        elif 'kill' in filename.lower() or 'switch' in filename.lower():
            expected_patterns = ['emergency_stop', 'kill_switch', 'force_close']
        else:
            expected_patterns = ['trade', 'order', 'position', 'signal']
        
        if not expected_patterns:
            return 50.0  # Default for unknown file types
        
        found_patterns = 0
        content_lower = content.lower()
        
        for pattern in expected_patterns:
            if pattern.lower() in content_lower:
                found_patterns += 1
        
        return (found_patterns / len(expected_patterns)) * 100.0
    
    def _assess_file_live_readiness(self, content: str) -> bool:
        """Assess if file is ready for live trading"""
        live_indicators = ['real_time', 'live_data', 'production', 'live_trading']
        demo_indicators = ['demo', 'test', 'mock', 'simulation', 'fake']
        
        content_lower = content.lower()
        
        has_live = any(indicator in content_lower for indicator in live_indicators)
        has_demo = any(indicator in content_lower for indicator in demo_indicators)
        
        # Must have live indicators and minimal demo/test references
        return has_live and not has_demo
    
    def _identify_file_issues(self, content: str, filename: str) -> List[str]:
        """Identify critical issues in trading file"""
        issues = []
        
        # Check for TODO/FIXME
        if re.search(r'(TODO|FIXME|HACK|XXX)', content, re.IGNORECASE):
            issues.append("Contains TODO/FIXME comments")
        
        # Check for hardcoded credentials
        if re.search(r'(password|key|secret)\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
            issues.append("Contains hardcoded credentials")
        
        # Check for error handling
        if 'try:' not in content and 'except' not in content:
            issues.append("Missing error handling")
        
        # File-specific checks
        if 'execution' in filename.lower():
            if not re.search(r'stop_loss|take_profit', content, re.IGNORECASE):
                issues.append("Missing SL/TP implementation")
        
        if 'risk' in filename.lower():
            if not re.search(r'position_size|risk_calc', content, re.IGNORECASE):
                issues.append("Missing position sizing logic")
        
        return issues
    
    def _assess_mt5_integration(self, found_files: List[Dict]) -> float:
        """Assess MT5 integration completeness"""
        mt5_features = ['connect', 'account_info', 'symbol_info', 'copy_rates', 'order_send', 'positions_get']
        found_features = 0
        
        for file_info in found_files:
            if 'mt5' in file_info['name'].lower() or 'connector' in file_info['name'].lower():
                try:
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for feature in mt5_features:
                        if feature in content:
                            found_features += 1
                except Exception:
                    continue
        
        return (found_features / len(mt5_features)) * 100.0
    
    def _assess_execution_paths(self, found_files: List[Dict]) -> float:
        """Assess execution path completeness"""
        execution_steps = ['signal_generation', 'risk_check', 'order_placement', 'position_monitoring', 'close_logic']
        found_steps = 0
        
        for file_info in found_files:
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                for step in execution_steps:
                    if any(keyword in content for keyword in step.split('_')):
                        found_steps += 1
                        break
            except Exception:
                continue
        
        return (found_steps / len(execution_steps)) * 100.0
    
    def _assess_risk_coverage(self, found_files: List[Dict]) -> float:
        """Assess risk management coverage"""
        risk_features = ['stop_loss', 'take_profit', 'position_size', 'max_drawdown', 'risk_limit']
        found_features = 0
        
        for file_info in found_files:
            if 'risk' in file_info['name'].lower() or 'guard' in file_info['name'].lower():
                try:
                    with open(file_info['path'], 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for feature in risk_features:
                        if feature in content:
                            found_features += 1
                except Exception:
                    continue
        
        return (found_features / len(risk_features)) * 100.0
    
    def _calculate_overall_completeness(self, analysis_results: List[Dict]) -> float:
        """Calculate overall trading intelligence completeness"""
        if not analysis_results:
            return 0.0
        
        analyzed_files = [f for f in analysis_results if f['analyzed']]
        if not analyzed_files:
            return 0.0
        
        total_completeness = sum(f['completeness'] for f in analyzed_files)
        return total_completeness / len(analyzed_files)
    
    def _calculate_live_readiness(self, analysis_results: List[Dict]) -> float:
        """Calculate live trading readiness score"""
        if not analysis_results:
            return 0.0
        
        analyzed_files = [f for f in analysis_results if f['analyzed']]
        if not analyzed_files:
            return 0.0
        
        live_ready_count = sum(1 for f in analyzed_files if f['live_ready'])
        return (live_ready_count / len(analyzed_files)) * 100.0
    
    def _generate_final_assessment(self, completeness: float, live_readiness: float, mt5_score: float):
        """Generate final assessment with issues and recommendations"""
        
        # Critical Issues
        if completeness < 70:
            self.critical_issues.append(f"Overall completeness too low: {completeness:.1f}% (target: 70%+)")
        
        if live_readiness < 50:
            self.critical_issues.append(f"Live readiness insufficient: {live_readiness:.1f}% (target: 50%+)")
        
        if mt5_score < 60:
            self.critical_issues.append(f"MT5 integration incomplete: {mt5_score:.1f}% (target: 60%+)")
        
        # Warnings
        if completeness < 85:
            self.warnings.append("Trading logic completeness could be improved")
        
        if live_readiness < 75:
            self.warnings.append("Live trading readiness needs enhancement")
        
        # Recommendations
        if completeness < 80:
            self.recommendations.append("Focus on completing core trading logic implementation")
        
        if live_readiness < 60:
            self.recommendations.append("Remove test/demo code and implement production-ready logic")
        
        if mt5_score < 80:
            self.recommendations.append("Complete MT5 integration with all required features")
        
        self.recommendations.append("Implement comprehensive error handling across all components")
        self.recommendations.append("Add extensive logging for production monitoring")

def generate_trading_audit_report(audit: TradingIntelligenceAudit, output_dir: str) -> Tuple[str, str]:
    """Generate optimized trading intelligence audit report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON Report
    json_path = os.path.join(output_dir, f"trading_intelligence_audit_report_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(asdict(audit), f, indent=2, default=str)
    
    # Markdown Report
    md_path = os.path.join(output_dir, f"TRADING_INTELLIGENCE_AUDIT_REPORT_{timestamp}.md")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"""# GENESIS TRADING INTELLIGENCE AUDIT REPORT
## Generated: {audit.timestamp}

---

## 🎯 EXECUTIVE SUMMARY

**Overall Trading Intelligence Completeness**: {audit.overall_completeness:.1f}%
**Live Trading Readiness Score**: {audit.live_readiness_score:.1f}%
**MT5 Integration Score**: {audit.mt5_integration_score:.1f}%
**Execution Paths Score**: {audit.execution_paths_score:.1f}%
**Risk Coverage Score**: {audit.risk_coverage_score:.1f}%

### 📊 ANALYSIS COVERAGE
- **Critical Files Found**: {audit.critical_files_found}
- **Files Successfully Analyzed**: {audit.critical_files_analyzed}
- **Analysis Coverage**: {(audit.critical_files_analyzed/audit.critical_files_found*100) if audit.critical_files_found > 0 else 0:.1f}%

---

## 🔥 SYSTEM STATUS ASSESSMENT

""")
        
        # Status determination
        if audit.overall_completeness >= 80 and audit.live_readiness_score >= 75:
            status = "🔥 PRODUCTION READY"
            status_color = "GREEN"
        elif audit.overall_completeness >= 60 and audit.live_readiness_score >= 50:
            status = "🚧 DEVELOPMENT STAGE"
            status_color = "YELLOW"
        else:
            status = "❌ INCOMPLETE"
            status_color = "RED"
        
        f.write(f"""
### Trading Intelligence Status: {status}

**ASSESSMENT**: 
- **Logic Completeness**: {'✅ Acceptable' if audit.overall_completeness >= 70 else '⚠️ Needs Work' if audit.overall_completeness >= 50 else '❌ Insufficient'}
- **Live Readiness**: {'✅ Ready' if audit.live_readiness_score >= 75 else '⚠️ Partial' if audit.live_readiness_score >= 50 else '❌ Not Ready'}
- **Integration Status**: {'✅ Good' if audit.mt5_integration_score >= 70 else '⚠️ Partial' if audit.mt5_integration_score >= 50 else '❌ Incomplete'}

---

## ⚠️ CRITICAL ISSUES ({len(audit.critical_issues)})
""")
        
        for issue in audit.critical_issues:
            f.write(f"- 🚨 {issue}\n")
        
        f.write(f"""
## ⚡ WARNINGS ({len(audit.warnings)})
""")
        for warning in audit.warnings:
            f.write(f"- ⚠️ {warning}\n")
        
        f.write(f"""
## 💡 RECOMMENDATIONS ({len(audit.recommendations)})
""")
        for rec in audit.recommendations:
            f.write(f"- 💡 {rec}\n")
        
        f.write(f"""
---

## 🎯 FINAL VERDICT

**LIVE TRADING READINESS**: {'🔥 APPROVED FOR LIVE TRADING' if audit.live_readiness_score >= 75 and audit.overall_completeness >= 80 else '🚧 REQUIRES DEVELOPMENT' if audit.overall_completeness >= 50 else '❌ NOT READY FOR LIVE TRADING'}

**IMMEDIATE NEXT ACTIONS**:
""")
        
        if audit.overall_completeness < 70:
            f.write("1. Complete core trading logic implementation\n")
        if audit.live_readiness_score < 60:
            f.write("2. Remove test/demo code, implement production logic\n")
        if audit.mt5_integration_score < 70:
            f.write("3. Complete MT5 integration implementation\n")
        
        f.write("""
4. Implement comprehensive error handling
5. Add production monitoring and logging
6. Conduct integration testing with live data feeds

---
*Generated by GENESIS Trading Intelligence Audit System v1.1.0 (Optimized)*
""")
    
    logger.info(f"📄 Trading Intelligence Audit Report saved to: {md_path}")
    logger.info(f"📄 Trading Intelligence Data saved to: {json_path}")
    
    return md_path, json_path

def main():
    """Main audit execution"""
    genesis_root = r"C:\Users\patra\Genesis FINAL TRY"
    
    print("🚀 GENESIS TRADING INTELLIGENCE AUDIT (OPTIMIZED)")
    print("=================================================")
    print(f"📁 Analyzing: {genesis_root}")
    print()
    
    # Initialize optimized auditor
    auditor = OptimizedTradingAuditor(genesis_root)
    
    # Execute comprehensive audit
    audit_result = auditor.run_comprehensive_audit()
    
    # Generate reports
    md_report, json_report = generate_trading_audit_report(audit_result, genesis_root)
    
    print()
    print("📊 AUDIT COMPLETE!")
    print(f"🎯 Overall Completeness: {audit_result.overall_completeness:.1f}%")
    print(f"🔥 Live Readiness: {audit_result.live_readiness_score:.1f}%") 
    print(f"⚡ MT5 Integration: {audit_result.mt5_integration_score:.1f}%")
    print(f"🛡️ Risk Coverage: {audit_result.risk_coverage_score:.1f}%")
    print()
    print(f"📄 Report: {md_report}")
    print(f"📄 Data: {json_report}")

if __name__ == "__main__":
    main()
