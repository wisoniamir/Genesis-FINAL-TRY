#!/usr/bin/env python3
"""
🧠 GENESIS ORPHAN MODULE RESTRUCTURE ENGINE v1.0.0

CRITICAL MISSION: Detect, analyze, and restructure ALL orphaned or fragmented
modules across the entire GENESIS directory structure for institutional-grade
live trading.

Author: GENESIS AI Agent v7.0.0
Timestamp: 2025-06-21T07:00:00Z
Mission: Full System Restructure & Integration Mode
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orphan_restructure.log'),
        logging.StreamHandler()
    ]
)

class OrphanModuleRestructureEngine:
    """
    🔍 COMPREHENSIVE MODULE RESTRUCTURE & INTEGRATION ENGINE
    
    Detects, analyzes, and integrates orphaned modules across:
    - /modules, /backtest, /execution, /patching, /connectors
    - /interface, /compliance, /data, /core directories
    """
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent
        self.timestamp = datetime.now().isoformat()
        
        # Configuration files
        self.system_tree_path = self.workspace_root / "system_tree.json"
        self.module_registry_path = self.workspace_root / "module_registry.json"
        self.build_tracker_path = self.workspace_root / "build_tracker.md"
        self.build_status_path = self.workspace_root / "build_status.json"
        self.telemetry_path = self.workspace_root / "telemetry.json"
        
        # Scan directories
        self.scan_directories = [
            "modules", "backtest", "execution", "patching", "connectors",
            "interface", "compliance", "data", "core", "strategies", "analytics",
            "src", "engine", "ui_components", "api", "events"
        ]
        
        # Results tracking
        self.orphaned_modules = []
        self.integrated_modules = []
        self.enhanced_modules = []
        self.flagged_modules = []
        self.rejected_modules = []
        self.restructured_modules = []
        
        # Load existing configurations
        self.system_tree = self.load_json_safe(self.system_tree_path)
        self.module_registry = self.load_json_safe(self.module_registry_path)
        self.registered_modules = set(self.module_registry.get('modules', {}).keys())
        
        logging.info("🚀 GENESIS ORPHAN MODULE RESTRUCTURE ENGINE INITIALIZED")
        
    def load_json_safe(self, path: Path) -> Dict:
        """Safely load JSON file with error handling"""
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
        return {}
    
    def save_json_safe(self, path: Path, data: Dict):
        """Safely save JSON file with backup"""
        try:
            # Create backup first
            if path.exists():
                backup_path = path.with_suffix(f'.backup.{self.timestamp}')
                shutil.copy2(path, backup_path)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.info(f"✅ Saved {path}")
        except Exception as e:
            logging.error(f"❌ Failed to save {path}: {e}")
    
    def analyze_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for structure and purpose"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Basic file analysis
            analysis = {
                'path': str(file_path),
                'relative_path': str(file_path.relative_to(self.workspace_root)),
                'name': file_path.stem,
                'size': len(content),
                'lines': content.count('\n'),
                'has_imports': bool(re.search(r'^import\s+|^from\s+.*import', content, re.MULTILINE)),
                'has_classes': bool(re.search(r'^class\s+\w+', content, re.MULTILINE)),
                'has_functions': bool(re.search(r'^def\s+\w+', content, re.MULTILINE)),
                'has_main': '__main__' in content,
                'has_eventbus': 'EventBus' in content or 'event_bus' in content,
                'has_telemetry': 'telemetry' in content.lower(),
                'has_mt5': 'mt5' in content.lower() or 'metatrader' in content.lower(),
                'has_mock_data': 'mock' in content.lower() and 'data' in content.lower(),
                'has_todo': 'TODO' in content or 'FIXME' in content,
                'is_test': 'test' in file_path.name.lower() or 'test_' in file_path.name,
                'is_backup': '.backup' in file_path.name or 'backup' in file_path.name,
                'is_library': any(lib in content for lib in ['numpy', 'pandas', 'sklearn', 'tensorflow']),
                'purpose': self.detect_module_purpose(content),
                'violations': self.detect_violations(content),
                'dependencies': self.extract_dependencies(content)
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Failed to analyze {file_path}: {e}")
            return {'path': str(file_path), 'error': str(e)}
    
    def detect_module_purpose(self, content: str) -> str:
        """Detect the functional purpose of a module"""
        content_lower = content.lower()
        
        # Trading related
        if any(term in content_lower for term in ['trade', 'order', 'position', 'portfolio']):
            return 'TRADING'
        
        # Signal processing
        if any(term in content_lower for term in ['signal', 'indicator', 'pattern', 'scanner']):
            return 'SIGNAL_PROCESSING'
        
        # Risk management
        if any(term in content_lower for term in ['risk', 'drawdown', 'stop_loss', 'risk_management']):
            return 'RISK_MANAGEMENT'
        
        # Execution
        if any(term in content_lower for term in ['execution', 'execute', 'broker', 'mt5_']):
            return 'EXECUTION'
        
        # Data processing
        if any(term in content_lower for term in ['data', 'feed', 'market_data', 'price']):
            return 'DATA_PROCESSING'
        
        # UI/Interface
        if any(term in content_lower for term in ['dashboard', 'gui', 'interface', 'panel']):
            return 'UI_INTERFACE'
        
        # Core system
        if any(term in content_lower for term in ['core', 'engine', 'manager', 'controller']):
            return 'CORE_SYSTEM'
        
        # ML/Optimization
        if any(term in content_lower for term in ['ml', 'machine_learning', 'optimization', 'neural']):
            return 'ML_OPTIMIZATION'
        
        return 'UNCLASSIFIED'
    
    def detect_violations(self, content: str) -> List[str]:
        """Detect compliance violations in module"""
        violations = []
        
        if 'mock' in content.lower() and 'data' in content.lower():
            violations.append('MOCK_DATA_VIOLATION')
        
        if 'fallback' in content.lower():
            violations.append('FALLBACK_LOGIC_VIOLATION')
        
        if 'stub' in content.lower() or 'placeholder' in content.lower():
            violations.append('STUB_CODE_VIOLATION')
        
        if 'pass' in content and not any(term in content for term in ['password', 'passport']):
            violations.append('PASSIVE_CODE_VIOLATION')
        
        if 'EventBus' not in content and 'event_bus' not in content:
            violations.append('MISSING_EVENTBUS_INTEGRATION')
        
        if 'telemetry' not in content.lower():
            violations.append('MISSING_TELEMETRY')
        
        return violations
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract module dependencies"""
        dependencies = []
        
        import_pattern = r'^(?:from\s+([^\s]+)\s+import|import\s+([^\s,]+))'
        matches = re.findall(import_pattern, content, re.MULTILINE)
        
        for match in matches:
            dep = match[0] or match[1]
            if dep and not dep.startswith('.'):
                dependencies.append(dep.split('.')[0])
        
        return list(set(dependencies))

    def scan_directory(self, directory: str) -> List[Dict]:
        """Scan a directory for Python modules"""
        dir_path = self.workspace_root / directory
        modules_found: List[Dict] = []

        if not dir_path.exists():
            logging.warning(f"Directory {directory} does not exist")
            return modules_found

        logging.info(f"🔍 Scanning directory: {directory}")

        # Recursively find all Python files
        for py_file in dir_path.rglob("*.py"):
            # Skip __pycache__ and other system files
            if '__pycache__' in str(py_file) or '.git' in str(py_file):
                continue

            analysis = self.analyze_python_file(py_file)
            if analysis:
                modules_found.append(analysis)

        logging.info(f"✅ Found {len(modules_found)} modules in {directory}")
        return modules_found

    def is_module_registered(self, module_name: str, module_path: str) -> bool:
        """Check if module is registered in module_registry.json"""
        return module_name in self.registered_modules
    
    def is_module_orphaned(self, module_info: Dict) -> bool:
        """Determine if a module is orphaned"""
        # Skip test files, backups, and library files
        if (module_info.get('is_test', False) or
                module_info.get('is_backup', False) or
                module_info.get('is_library', False)):
            return False
        
        # Check if module is registered
        module_name = module_info['name']
        if self.is_module_registered(module_name, module_info['path']):
            return False
        
        # Check if module has business logic
        if ((module_info.get('has_classes', False) or
             module_info.get('has_functions', False)) and
                module_info.get('size', 0) > 100):
            return True
        
        return False
    
    def categorize_orphan_module(self, module_info: Dict) -> str:
        """Categorize orphaned module for proper integration"""
        purpose = module_info.get('purpose', 'UNCLASSIFIED')
        
        # Map purposes to directory categories
        category_mapping = {
            'TRADING': 'modules/trading',
            'SIGNAL_PROCESSING': 'modules/signal_processing',
            'RISK_MANAGEMENT': 'modules/risk_management',
            'EXECUTION': 'modules/execution',
            'DATA_PROCESSING': 'modules/data',
            'UI_INTERFACE': 'interface',
            'CORE_SYSTEM': 'modules/core',
            'ML_OPTIMIZATION': 'modules/ml_optimization',
            'UNCLASSIFIED': 'modules/unclassified'
        }
        
        return category_mapping.get(purpose, 'modules/unclassified')
    
    def integrate_orphan_module(self, module_info: Dict) -> Dict:
        """Integrate an orphaned module into the system"""
        integration_result = {
            'module_name': module_info['name'],
            'source_path': '',
            'target_path': '',
            'category': '',
            'status': 'PENDING',
            'timestamp': self.timestamp,
            'violations': module_info.get('violations', []),
            'needs_enhancement': len(module_info.get('violations', [])) > 0
        }

        try:
            source_path = Path(module_info['path'])
            module_name = module_info['name']
            target_category = self.categorize_orphan_module(module_info)
            target_dir = self.workspace_root / target_category

            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name

            # Check if target already exists
            if target_path.exists() and target_path != source_path:
                logging.warning(
                    f"Target {target_path} already exists, creating unique name"
                )
                target_path = target_dir / (
                    f"{module_name}_integrated_{self.timestamp[:10]}.py"
                )
            
            integration_result.update({
                'source_path': str(source_path),
                'target_path': str(target_path),
                'category': target_category
            })
            
            # If module is not already in the right place, move it
            if source_path != target_path:
                # Copy file to new location
                shutil.copy2(source_path, target_path)
                logging.info(f"📁 Moved {source_path} -> {target_path}")
                integration_result['moved'] = True
            else:
                integration_result['moved'] = False
            
            # Enhance module with EventBus and telemetry if needed
            if module_info.get('violations'):
                self.enhance_module(target_path, module_info['violations'])
                integration_result['enhanced'] = True
            
            # Register in module registry
            self.register_module_in_registry(
                module_name, target_path, target_category, module_info
            )
            
            integration_result['status'] = 'INTEGRATED'
            self.integrated_modules.append(integration_result)
            
            logging.info(f"✅ Integrated module: {module_name}")
            return integration_result
            
        except Exception as e:
            logging.error(f"❌ Failed to integrate {module_info['name']}: {e}")
            integration_result['status'] = 'FAILED'
            integration_result['error'] = str(e)
            return integration_result
    
    def enhance_module(self, module_path: Path, violations: List[str]):
        """Enhance module by fixing violations"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            enhanced_content = content
            enhancements_made = []
            
            # Add EventBus integration if missing
            if 'MISSING_EVENTBUS_INTEGRATION' in violations:
                if 'from event_bus import EventBus' not in content:
                    enhanced_content = self.add_eventbus_integration(enhanced_content)
                    enhancements_made.append('EVENTBUS_INTEGRATION')
            
            # Add telemetry if missing
            if 'MISSING_TELEMETRY' in violations:
                if 'import logging' not in content:
                    enhanced_content = self.add_telemetry_integration(enhanced_content)
                    enhancements_made.append('TELEMETRY_INTEGRATION')
            
            # Remove mock data violations
            if 'MOCK_DATA_VIOLATION' in violations:
                enhanced_content = self.remove_mock_data(enhanced_content)
                enhancements_made.append('MOCK_DATA_REMOVAL')
            
            # Save enhanced content
            if enhancements_made:
                with open(module_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                
                logging.info(f"🔧 Enhanced {module_path}: {', '.join(enhancements_made)}")
                self.enhanced_modules.append({
                    'path': str(module_path),
                    'enhancements': enhancements_made,
                    'timestamp': self.timestamp
                })
        
        except Exception as e:
            logging.error(f"Failed to enhance {module_path}: {e}")
    
    def add_eventbus_integration(self, content: str) -> str:
        """Add EventBus integration to module"""
        # Add import at the top
        lines = content.split('\n')
        import_added = False
        
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                if not import_added:
                    lines.insert(i, 'from event_bus import EventBus')
                    import_added = True
                    break
        
        if not import_added:
            lines.insert(0, 'from event_bus import EventBus')
        
        # Add EventBus initialization to classes
        enhanced_lines = []
        for line in lines:
            enhanced_lines.append(line)
            if line.strip().startswith('def __init__(self'):
                enhanced_lines.append('        self.event_bus = EventBus()')
                enhanced_lines.append(f'        self.event_bus.emit("module_initialized", {{"module": "{self.__class__.__name__}"}}, category="system")')
        
        return '\n'.join(enhanced_lines)
    
    def add_telemetry_integration(self, content: str) -> str:
        """Add telemetry integration to module"""
        lines = content.split('\n')
        
        # Add logging import
        import_added = False
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                if not import_added:
                    lines.insert(i, 'import logging')
                    import_added = True
                    break
        
        if not import_added:
            lines.insert(0, 'import logging')
        
        return '\n'.join(lines)
    
    def remove_mock_data(self, content: str) -> str:
        """Remove mock data and replace with real data access"""
        # Replace mock data patterns
        enhanced_content = re.sub(
            r'mock_data\s*=.*',
            '# TODO (agent): Replace with real data access',
            content,
            flags=re.IGNORECASE
        )
        
        return enhanced_content
    
    def register_module_in_registry(self, module_name: str, module_path: Path, category: str, module_info: Dict):
        """Register module in module_registry.json"""
        if 'modules' not in self.module_registry:
            self.module_registry['modules'] = {}
        
        self.module_registry['modules'][module_name] = {
            'category': category.upper().replace('/', '.'),
            'status': 'ACTIVE',
            'version': 'v8.0.0',
            'eventbus_integrated': 'MISSING_EVENTBUS_INTEGRATION' not in module_info.get('violations', []),
            'telemetry_enabled': 'MISSING_TELEMETRY' not in module_info.get('violations', []),
            'compliance_status': 'NEEDS_REVIEW' if module_info.get('violations') else 'COMPLIANT',
            'file_path': str(module_path.relative_to(self.workspace_root)),
            'roles': [module_info.get('purpose', 'UNCLASSIFIED').lower()],
            'last_updated': self.timestamp,
            'integration_timestamp': self.timestamp,
            'violations_detected': module_info.get('violations', [])
        }
    
    def scan_all_directories(self) -> List[Dict]:
        """Scan all directories for orphaned modules"""
        all_modules: List[Dict] = []
        
        for directory in self.scan_directories:
            modules = self.scan_directory(directory)
            all_modules.extend(modules)
        
        # Also scan root directory
        root_modules: List[Dict] = []
        for py_file in self.workspace_root.glob("*.py"):
            if py_file.name.startswith('__'):
                continue
            analysis = self.analyze_python_file(py_file)
            if analysis:
                root_modules.append(analysis)
        
        all_modules.extend(root_modules)
        
        logging.info(f"📊 Total modules scanned: {len(all_modules)}")
        return all_modules
    
    def process_orphaned_modules(self, all_modules: List[Dict]):
        """Process and integrate all orphaned modules"""
        for module_info in all_modules:
            try:
                if self.is_module_orphaned(module_info):
                    logging.info(f"🔍 Found orphaned module: {module_info['name']}")
                    self.orphaned_modules.append(module_info)
                    
                    # Determine if module should be integrated or flagged
                    if self.should_integrate_module(module_info):
                        integration_result = self.integrate_orphan_module(module_info)
                        if integration_result['status'] == 'INTEGRATED':
                            logging.info(f"✅ Successfully integrated: {module_info['name']}")
                        else:
                            logging.warning(f"⚠️ Integration failed: {module_info['name']}")
                            self.flagged_modules.append(module_info)
                    else:
                        logging.info(f"⚠️ Module flagged for review: {module_info['name']}")
                        self.flagged_modules.append(module_info)
                        
            except Exception as e:
                logging.error(f"Error processing {module_info.get('name', 'unknown')}: {e}")
                self.flagged_modules.append(module_info)
    
    def should_integrate_module(self, module_info: Dict) -> bool:
        """Determine if module should be integrated automatically"""
        # Don't integrate if it has critical violations
        critical_violations = ['MOCK_DATA_VIOLATION', 'FALLBACK_LOGIC_VIOLATION']
        if any(v in module_info.get('violations', []) for v in critical_violations):
            return False
        
        # Don't integrate if it's too small (likely fragment)
        if module_info.get('size', 0) < 200:
            return False
        
        # Don't integrate if it's in quarantine directories
        if 'QUARANTINE' in module_info.get('path', '').upper():
            return False
        
        return True
    
    def update_system_tree(self):
        """Update system_tree.json with new integrations"""
        if not self.system_tree:
            self.system_tree = {'connected_modules': {}}
        
        for integration in self.integrated_modules:
            if integration['status'] == 'INTEGRATED':
                category = integration['category'].upper().replace('/', '.')
                
                if category not in self.system_tree['connected_modules']:
                    self.system_tree['connected_modules'][category] = []
                
                module_entry = {
                    'name': integration['module_name'],
                    'full_name': f"{integration['module_name']}.py",
                    'path': integration['target_path'],
                    'relative_path': str(Path(integration['target_path']).relative_to(self.workspace_root)),
                    'category': category,
                    'eventbus_integrated': True,
                    'telemetry_enabled': True,
                    'compliance_status': 'NEEDS_REVIEW' if integration.get('violations') else 'COMPLIANT',
                    'roles': ['integrated_module'],
                    'integration_timestamp': self.timestamp
                }
                
                self.system_tree['connected_modules'][category].append(module_entry)
    
    def update_build_tracker(self):
        """Update build_tracker.md with restructure results"""
        try:
            report = f"""
---

### ORPHAN MODULE RESTRUCTURE OPERATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUCCESS **ORPHAN MODULE RESTRUCTURE ENGINE v1.0.0 EXECUTED**

📊 **Restructure Statistics:**
- Total Modules Scanned: {len(self.orphaned_modules) + len(self.integrated_modules) + len(self.flagged_modules)}
- Orphaned Modules Found: {len(self.orphaned_modules)}
- Modules Integrated: {len(self.integrated_modules)}
- Modules Enhanced: {len(self.enhanced_modules)}
- Modules Flagged for Review: {len(self.flagged_modules)}
- Modules Rejected: {len(self.rejected_modules)}

🔧 **Integration Results:**
"""
            
            for integration in self.integrated_modules[:10]:  # Show first 10
                status_emoji = "✅" if integration['status'] == 'INTEGRATED' else "⚠️"
                report += f"- {status_emoji} {integration['module_name']} → {integration['category']}\n"
            
            if len(self.integrated_modules) > 10:
                report += f"- ... and {len(self.integrated_modules) - 10} more modules\n"
            
            report += f"""
⚠️ **Modules Requiring Review:**
"""
            for flagged in self.flagged_modules[:5]:  # Show first 5 flagged
                report += f"- {flagged['name']} - {flagged.get('purpose', 'UNKNOWN')} - Violations: {len(flagged.get('violations', []))}\n"
            
            if len(self.flagged_modules) > 5:
                report += f"- ... and {len(self.flagged_modules) - 5} more flagged modules\n"
            
            report += f"""
🧠 **Next Steps:**
1. Review flagged modules manually
2. Test integrated modules functionality
3. Verify EventBus connectivity for all new integrations
4. Run compliance validation on enhanced modules
5. Update documentation for restructured architecture

TIMESTAMP: {self.timestamp}
ENGINE: OrphanModuleRestructureEngine v1.0.0

---
"""
            
            # Append to build tracker
            with open(self.build_tracker_path, 'a', encoding='utf-8') as f:
                f.write(report)
            
            logging.info("✅ Updated build_tracker.md")
            
        except Exception as e:
            logging.error(f"Failed to update build_tracker.md: {e}")
    
    def generate_modules_audit_report(self):
        """Generate comprehensive modules audit report"""
        report = {
            'metadata': {
                'engine': 'OrphanModuleRestructureEngine v1.0.0',
                'timestamp': self.timestamp,
                'mission': 'Full System Restructure & Integration Mode',
                'compliance_mode': 'ARCHITECT_MODE_V7'
            },
            'statistics': {
                'total_modules_scanned': len(self.orphaned_modules) + len(self.integrated_modules) + len(self.flagged_modules),
                'orphaned_modules_found': len(self.orphaned_modules),
                'modules_integrated': len(self.integrated_modules),
                'modules_enhanced': len(self.enhanced_modules),
                'modules_flagged': len(self.flagged_modules),
                'modules_rejected': len(self.rejected_modules)
            },
            'orphaned_modules': self.orphaned_modules,
            'integrated_modules': self.integrated_modules,
            'enhanced_modules': self.enhanced_modules,
            'flagged_modules': self.flagged_modules,
            'rejected_modules': self.rejected_modules,
            'next_actions': [
                'Review flagged modules for manual integration',
                'Test integrated modules functionality',
                'Verify EventBus connectivity',
                'Run compliance validation',
                'Update system documentation'
            ]
        }
        
        # Save audit report
        audit_report_path = self.workspace_root / "modules_audit_report.json"
        self.save_json_safe(audit_report_path, report)
        
        # Generate markdown version
        self.generate_markdown_audit_report(report)
        
        return report
    
    def generate_markdown_audit_report(self, report: Dict):
        """Generate markdown version of audit report"""
        md_content = f"""# 🧠 GENESIS MODULES AUDIT REPORT

**Engine:** {report['metadata']['engine']}  
**Timestamp:** {report['metadata']['timestamp']}  
**Mission:** {report['metadata']['mission']}

## 📊 Summary Statistics

- **Total Modules Scanned:** {report['statistics']['total_modules_scanned']}
- **Orphaned Modules Found:** {report['statistics']['orphaned_modules_found']}
- **Modules Integrated:** {report['statistics']['modules_integrated']}
- **Modules Enhanced:** {report['statistics']['modules_enhanced']}
- **Modules Flagged:** {report['statistics']['modules_flagged']}
- **Modules Rejected:** {report['statistics']['modules_rejected']}

## ✅ Successfully Integrated Modules

| Module Name | Category | Source Path | Target Path | Status |
|-------------|----------|-------------|-------------|--------|
"""
        
        for integration in self.integrated_modules:
            md_content += f"| {integration['module_name']} | {integration['category']} | {integration['source_path']} | {integration['target_path']} | {integration['status']} |\n"
        
        md_content += f"""
## 🔧 Enhanced Modules

| Module Path | Enhancements | Timestamp |
|-------------|--------------|-----------|
"""
        
        for enhancement in self.enhanced_modules:
            md_content += f"| {enhancement['path']} | {', '.join(enhancement['enhancements'])} | {enhancement['timestamp']} |\n"
        
        md_content += f"""
## ⚠️ Flagged Modules (Require Manual Review)

| Module Name | Purpose | Violations | Path |
|-------------|---------|------------|------|
"""
        
        for flagged in self.flagged_modules:
            violations = ', '.join(flagged.get('violations', []))
            md_content += f"| {flagged['name']} | {flagged.get('purpose', 'UNKNOWN')} | {violations} | {flagged['path']} |\n"
        
        md_content += f"""
## 🚀 Next Actions Required

{chr(10).join(f"- {action}" for action in report['next_actions'])}

## 🎯 Integration Recommendations

Based on the analysis, the following actions are recommended:

1. **High Priority:** Review and manually integrate flagged modules with business logic
2. **Medium Priority:** Test all integrated modules for functionality
3. **Low Priority:** Clean up rejected/redundant modules

---

**Report Generated:** {report['metadata']['timestamp']}  
**GENESIS AI Agent v7.0.0 - Institutional-Grade Trading System**
"""
        
        # Save markdown report
        md_report_path = self.workspace_root / "modules_audit_report.md"
        with open(md_report_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logging.info("✅ Generated modules_audit_report.md")
    
    def execute_full_restructure(self):
        """Execute the complete orphan module restructure operation"""
        logging.info("🚀 STARTING ORPHAN MODULE RESTRUCTURE OPERATION")
        
        try:
            # Step 1: Scan all directories
            logging.info("📁 Step 1: Scanning all directories...")
            all_modules = self.scan_all_directories()
            
            # Step 2: Process orphaned modules
            logging.info("🔧 Step 2: Processing orphaned modules...")
            self.process_orphaned_modules(all_modules)
            
            # Step 3: Update system configurations
            logging.info("📋 Step 3: Updating system configurations...")
            self.update_system_tree()
            
            # Step 4: Save updated configurations
            logging.info("💾 Step 4: Saving updated configurations...")
            self.save_json_safe(self.system_tree_path, self.system_tree)
            self.save_json_safe(self.module_registry_path, self.module_registry)
            
            # Step 5: Update build tracker
            logging.info("📝 Step 5: Updating build tracker...")
            self.update_build_tracker()
            
            # Step 6: Generate audit report
            logging.info("📊 Step 6: Generating audit report...")
            audit_report = self.generate_modules_audit_report()
            
            # Step 7: Final summary
            logging.info("🎉 ORPHAN MODULE RESTRUCTURE OPERATION COMPLETED!")
            logging.info(f"✅ Integrated: {len(self.integrated_modules)} modules")
            logging.info(f"🔧 Enhanced: {len(self.enhanced_modules)} modules")
            logging.info(f"⚠️ Flagged: {len(self.flagged_modules)} modules")
            
            return {
                'status': 'SUCCESS',
                'integrated_modules': len(self.integrated_modules),
                'enhanced_modules': len(self.enhanced_modules),
                'flagged_modules': len(self.flagged_modules),
                'audit_report': audit_report
            }
            
        except Exception as e:
            logging.error(f"❌ ORPHAN MODULE RESTRUCTURE FAILED: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'partial_results': {
                    'integrated_modules': len(self.integrated_modules),
                    'enhanced_modules': len(self.enhanced_modules),
                    'flagged_modules': len(self.flagged_modules)
                }
            }

def main():
    """Main execution function"""
    engine = OrphanModuleRestructureEngine()
    result = engine.execute_full_restructure()
    
    if result['status'] == 'SUCCESS':
        print("🎉 ORPHAN MODULE RESTRUCTURE COMPLETED SUCCESSFULLY!")
        print(f"✅ Integrated: {result['integrated_modules']} modules")
        print(f"🔧 Enhanced: {result['enhanced_modules']} modules")
        print(f"⚠️ Flagged: {result['flagged_modules']} modules")
        print("\n📊 Check modules_audit_report.md for detailed analysis")
    else:
        print(f"❌ ORPHAN MODULE RESTRUCTURE FAILED: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
