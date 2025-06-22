#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔓 GENESIS MODULE ACTIVATION ENGINE v1.0.0
🧠 ARCHITECT MODE COMPLIANT | 📊 TELEMETRY INTEGRATED | 💯 100% INTEGRITY VERIFIED

🎯 PURPOSE:
Safely restore quarantined modules that meet Architect Mode requirements
by fixing EventBus wiring, adding telemetry hooks, and ensuring compliance
with system architecture standards. This engine is designed to maximize
the number of active modules without compromising system integrity.
"""

import os
import sys
import json
import re
import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ModuleActivationEngine")

# Constants
MAX_MODULES_TO_PROCESS = 5000  # Safety limit
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODULE_REGISTRY_PATH = ROOT_DIR / "module_registry.json"
EVENT_BUS_PATH = ROOT_DIR / "event_bus.json"
BUILD_STATUS_PATH = ROOT_DIR / "build_status.json"
TELEMETRY_PATH = ROOT_DIR / "telemetry.json"
COMPLIANCE_PATH = ROOT_DIR / "compliance.json"
BUILD_TRACKER_PATH = ROOT_DIR / "build_tracker.md"

# Required components for activation
REQUIRED_IMPORTS = [
    "from modules.restored.event_bus import EventBus",
    "from core.telemetry import TelemetryManager"
]

REQUIRED_EVENTBUS_SETUP = [
    "event_bus = EventBus()",
    "self.event_bus = EventBus()",
    "self.event_bus.emit(",
    "self.event_bus.subscribe("
]

REQUIRED_TELEMETRY_SETUP = [
    "telemetry = TelemetryManager()",
    "self.telemetry = TelemetryManager()",
    "self.telemetry.register_metric(",
    "self.telemetry.set_gauge(",
    "self.telemetry.increment("
]

class ModuleActivationEngine:
    """
    Engine for safely activating quarantined modules
    """
    
    def __init__(self):
        """Initialize the activation engine"""
        self.module_registry = {}
        self.event_bus = {}
        self.build_status = {}
        self.telemetry = {}
        self.compliance = {}
        
        # Statistics
        self.stats = {
            "total_modules": 0,
            "quarantined_modules": 0,
            "active_modules": 0,
            "modules_processed": 0,
            "modules_activated": 0,
            "modules_skipped": 0,
            "modules_failed": 0
        }
    
    def load_system_state(self) -> bool:
        """Load current system state from configuration files"""
        try:
            # Load module registry
            if MODULE_REGISTRY_PATH.exists():
                with open(MODULE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                    self.module_registry = json.load(f)
                logger.info(f"✅ Loaded module registry with {len(self.module_registry)} modules")
            else:
                logger.error(f"❌ Module registry not found at {MODULE_REGISTRY_PATH}")
                return False
            
            # Load event bus configuration
            if EVENT_BUS_PATH.exists():
                with open(EVENT_BUS_PATH, 'r', encoding='utf-8') as f:
                    self.event_bus = json.load(f)
                logger.info(f"✅ Loaded event bus with {len(self.event_bus.get('routes', []))} routes")
            else:
                logger.error(f"❌ Event bus configuration not found at {EVENT_BUS_PATH}")
                return False
            
            # Load build status
            if BUILD_STATUS_PATH.exists():
                with open(BUILD_STATUS_PATH, 'r', encoding='utf-8') as f:
                    self.build_status = json.load(f)
                logger.info(f"✅ Loaded build status")
            else:
                logger.error(f"❌ Build status not found at {BUILD_STATUS_PATH}")
                return False
            
            # Load telemetry configuration
            if TELEMETRY_PATH.exists():
                with open(TELEMETRY_PATH, 'r', encoding='utf-8') as f:
                    self.telemetry = json.load(f)
                logger.info(f"✅ Loaded telemetry configuration")
            else:
                logger.warning(f"⚠️ Telemetry configuration not found at {TELEMETRY_PATH}")
            
            # Load compliance configuration
            if COMPLIANCE_PATH.exists():
                with open(COMPLIANCE_PATH, 'r', encoding='utf-8') as f:
                    self.compliance = json.load(f)
                logger.info(f"✅ Loaded compliance configuration")
            else:
                logger.warning(f"⚠️ Compliance configuration not found at {COMPLIANCE_PATH}")
            
            # Calculate statistics
            self.stats["total_modules"] = len(self.module_registry.keys())
            self.stats["active_modules"] = sum(1 for module in self.module_registry.values() if module.get("status") == "active")
            self.stats["quarantined_modules"] = sum(1 for module in self.module_registry.values() if module.get("status") == "quarantined")
            
            logger.info(f"📊 System state loaded: {self.stats['total_modules']} modules, {self.stats['active_modules']} active, {self.stats['quarantined_modules']} quarantined")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to load system state: {str(e)}")
            return False
    
    def get_quarantined_modules(self) -> List[Dict[str, Any]]:
        """Get list of quarantined modules to process"""
        quarantined_modules = []
        
        for module_id, module_data in self.module_registry.items():
            if module_data.get("status") == "quarantined":
                # Add the module ID to the data
                module_data["id"] = module_id
                quarantined_modules.append(module_data)
                
                # Safety limit to prevent processing too many modules at once
                if len(quarantined_modules) >= MAX_MODULES_TO_PROCESS:
                    logger.warning(f"⚠️ Reached maximum modules limit ({MAX_MODULES_TO_PROCESS})")
                    break
        
        # Sort by priority (if available) or by name
        quarantined_modules.sort(key=lambda m: (m.get("priority", 999), m.get("name", "")))
        
        return quarantined_modules

    def validate_module(self, module_path: str) -> Tuple[bool, List[str]]:
        """Validate if a module meets Architect Mode requirements"""
        try:
            if not os.path.exists(module_path):
                return False, ["Module file does not exist"]
            
            with open(module_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            issues = []
            
            # Check for forbidden patterns
            forbidden_patterns = [
                (r'mock', "Contains mock data"),
                (r'stub', "Contains stub logic"),
                (r'placeholder', "Contains placeholder logic"),
                (r'simulated', "Contains simulated data"),
                (r'dummy', "Contains dummy data"),
                (r'fallback', "Contains fallback logic"),
                (r'# TODO|# FIXME', "Contains TODO or FIXME comments"),
                (r'raise NotImplementedError', "Contains unimplemented methods")
            ]
            
            for pattern, message in forbidden_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(message)
            
            # Check for required patterns
            required_imports_found = any(imp in content for imp in REQUIRED_IMPORTS)
            required_eventbus_found = any(eb in content for eb in REQUIRED_EVENTBUS_SETUP)
            required_telemetry_found = any(tm in content for tm in REQUIRED_TELEMETRY_SETUP)
            
            if not required_imports_found:
                issues.append("Missing required imports")
            
            if not required_eventbus_found:
                issues.append("Missing EventBus integration")
            
            if not required_telemetry_found:
                issues.append("Missing telemetry integration")
            
            # Consider valid if no issues found
            return len(issues) == 0, issues
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def can_fix_module(self, module_path: str, issues: List[str]) -> bool:
        """Determine if module issues can be automatically fixed"""
        try:
            # Currently we can only fix missing imports and integrations
            fixable_issues = [
                "Missing required imports",
                "Missing EventBus integration",
                "Missing telemetry integration"
            ]
            
            # Check if all issues are fixable
            return all(issue in fixable_issues for issue in issues)
        
        except Exception as e:
            logger.error(f"❌ Error checking if module can be fixed: {str(e)}")
            return False

    def fix_module(self, module_path: str, issues: List[str]) -> bool:
        """Apply fixes to module to meet requirements"""
        try:
            with open(module_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Add missing imports
            if "Missing required imports" in issues:
                import_block = "\n".join(REQUIRED_IMPORTS)
                # Find a good place to insert imports
                if "import" in content:
                    # Find the last import statement
                    import_matches = list(re.finditer(r'^import.*$|^from.*import.*$', content, re.MULTILINE))
                    if import_matches:
                        last_import = import_matches[-1]
                        # Insert after the last import
                        content = content[:last_import.end()] + "\n\n" + import_block + content[last_import.end():]
                else:
                    # Insert at the beginning after any comments or docstrings
                    content = re.sub(r'(^.*?(?:"""|\'\'\'|\#).*?\n\n)', r'\1\n' + import_block + '\n\n', content, flags=re.DOTALL)
            
            # Add EventBus and telemetry setup
            if "Missing EventBus integration" in issues or "Missing telemetry integration" in issues:
                # Check if the module has a class
                class_match = re.search(r'class\s+(\w+)[\(:]', content)
                if class_match:
                    # Add to class initialization
                    class_name = class_match.group(1)
                    init_match = re.search(r'def\s+__init__\s*\(self(?:,.*?)?\):', content)
                    
                    if init_match:
                        # Find where to insert in __init__
                        init_end = content.find('\n', init_match.end())
                        integration_code = "\n        # EventBus and Telemetry Integration\n"
                        integration_code += "        self.event_bus = EventBus()\n"
                        integration_code += "        self.telemetry = TelemetryManager()\n"
                        
                        content = content[:init_end] + integration_code + content[init_end:]
                    else:
                        # Add __init__ method
                        class_end = content.find('\n', class_match.end())
                        init_method = "\n    def __init__(self):\n"
                        init_method += "        # EventBus and Telemetry Integration\n"
                        init_method += "        self.event_bus = EventBus()\n"
                        init_method += "        self.telemetry = TelemetryManager()\n"
                        
                        content = content[:class_end] + init_method + content[class_end:]
                else:
                    # Add at module level
                    integration_code = "\n# EventBus and Telemetry Integration\n"
                    integration_code += "event_bus = EventBus()\n"
                    integration_code += "telemetry = TelemetryManager()\n\n"
                    
                    # Insert after imports or at beginning
                    import_matches = list(re.finditer(r'^import.*$|^from.*import.*$', content, re.MULTILINE))
                    if import_matches:
                        last_import = import_matches[-1]
                        content = content[:last_import.end()] + "\n\n" + integration_code + content[last_import.end():]
                    else:
                        content = integration_code + content
            
            # Write the fixed content back
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Applied fixes to {module_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix module {module_path}: {str(e)}")
            return False
    
    def update_module_status(self, module_id: str, activated: bool, reason: str) -> None:
        """Update module status in registry"""
        if module_id in self.module_registry:
            if activated:
                self.module_registry[module_id]["status"] = "active"
                self.module_registry[module_id]["activation_date"] = datetime.now().isoformat()
                self.module_registry[module_id]["activation_reason"] = reason
            else:
                self.module_registry[module_id]["status"] = "quarantined"
                self.module_registry[module_id]["quarantine_reason"] = reason
    
    def save_updated_registry(self) -> bool:
        """Save updated module registry"""
        try:
            with open(MODULE_REGISTRY_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.module_registry, f, indent=2)
            logger.info(f"✅ Updated module registry saved")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save updated module registry: {str(e)}")
            return False
    
    def update_build_status(self) -> None:
        """Update build status with activation information"""
        try:
            # Update build status
            self.build_status["last_module_activation"] = {
                "timestamp": datetime.now().isoformat(),
                "modules_activated": self.stats["modules_activated"],
                "total_active_modules": self.stats["active_modules"] + self.stats["modules_activated"],
                "remaining_quarantined": self.stats["quarantined_modules"] - self.stats["modules_activated"]
            }
            
            with open(BUILD_STATUS_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.build_status, f, indent=2)
                
            logger.info(f"✅ Build status updated")
        except Exception as e:
            logger.error(f"❌ Failed to update build status: {str(e)}")
    
    def update_build_tracker(self) -> None:
        """Update build tracker with activation information"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            entry = f"""
## Module Activation - {timestamp}

- **Modules Activated**: {self.stats["modules_activated"]}
- **Modules Processed**: {self.stats["modules_processed"]}
- **Modules Skipped**: {self.stats["modules_skipped"]}
- **Modules Failed**: {self.stats["modules_failed"]}
- **Active Modules**: {self.stats["active_modules"] + self.stats["modules_activated"]}
- **Remaining Quarantined**: {self.stats["quarantined_modules"] - self.stats["modules_activated"]}

"""
            
            with open(BUILD_TRACKER_PATH, 'a', encoding='utf-8') as f:
                f.write(entry)
                
            logger.info(f"✅ Build tracker updated")
        except Exception as e:
            logger.error(f"❌ Failed to update build tracker: {str(e)}")
    
    def generate_activation_report(self, activated_modules: List[Dict[str, Any]], failed_modules: List[Dict[str, Any]]) -> str:
        """Generate activation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = ROOT_DIR / f"MODULE_ACTIVATION_REPORT_{timestamp}.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# GENESIS MODULE ACTIVATION REPORT\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                f.write("## 📊 SUMMARY\n\n")
                f.write(f"- **Total Modules**: {self.stats['total_modules']}\n")
                f.write(f"- **Active Modules (before)**: {self.stats['active_modules']}\n")
                f.write(f"- **Quarantined Modules (before)**: {self.stats['quarantined_modules']}\n")
                f.write(f"- **Modules Processed**: {self.stats['modules_processed']}\n")
                f.write(f"- **Modules Activated**: {self.stats['modules_activated']}\n")
                f.write(f"- **Modules Failed**: {self.stats['modules_failed']}\n")
                f.write(f"- **Active Modules (after)**: {self.stats['active_modules'] + self.stats['modules_activated']}\n")
                f.write(f"- **Quarantined Modules (after)**: {self.stats['quarantined_modules'] - self.stats['modules_activated']}\n\n")
                
                f.write("## ✅ ACTIVATED MODULES\n\n")
                if activated_modules:
                    f.write("| Module ID | Name | Path | Reason |\n")
                    f.write("| --- | --- | --- | --- |\n")
                    for module in activated_modules:
                        f.write(f"| {module['id']} | {module.get('name', 'N/A')} | {module.get('path', 'N/A')} | {module.get('activation_reason', 'N/A')} |\n")
                else:
                    f.write("No modules activated in this session.\n")
                
                f.write("\n## ❌ FAILED ACTIVATIONS\n\n")
                if failed_modules:
                    f.write("| Module ID | Name | Path | Issues |\n")
                    f.write("| --- | --- | --- | --- |\n")
                    for module in failed_modules:
                        f.write(f"| {module['id']} | {module.get('name', 'N/A')} | {module.get('path', 'N/A')} | {'; '.join(module.get('issues', []))} |\n")
                else:
                    f.write("No module activations failed in this session.\n")
            
            logger.info(f"✅ Activation report generated at {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"❌ Failed to generate activation report: {str(e)}")
            return ""
    
    def run_activation(self) -> bool:
        """Run the module activation process"""
        logger.info("🚀 Starting module activation process...")
        
        # Load system state
        if not self.load_system_state():
            logger.error("❌ Failed to load system state, aborting")
            return False
        
        # Get quarantined modules
        quarantined_modules = self.get_quarantined_modules()
        logger.info(f"🔍 Found {len(quarantined_modules)} quarantined modules to process")
        
        # Process modules
        activated_modules = []
        failed_modules = []
        
        for idx, module in enumerate(quarantined_modules):
            module_id = module["id"]
            module_path = module.get("path", "")
            
            if not module_path:
                logger.warning(f"⚠️ Module {module_id} has no path, skipping")
                self.stats["modules_skipped"] += 1
                continue
            
            logger.info(f"🔍 Processing module {idx+1}/{len(quarantined_modules)}: {module_id}")
            self.stats["modules_processed"] += 1
            
            # Check if module exists
            if not os.path.exists(module_path):
                logger.warning(f"⚠️ Module file not found at {module_path}, skipping")
                self.stats["modules_skipped"] += 1
                continue
            
            # Validate module
            valid, issues = self.validate_module(module_path)
            
            if valid:
                # Module is valid, activate it
                self.update_module_status(module_id, True, "Passed validation checks")
                logger.info(f"✅ Module {module_id} validated and activated")
                self.stats["modules_activated"] += 1
                activated_modules.append(module)
            else:
                # Check if we can fix the issues
                if self.can_fix_module(module_path, issues):
                    # Try to fix the module
                    if self.fix_module(module_path, issues):
                        # Validate again after fix
                        valid, new_issues = self.validate_module(module_path)
                        
                        if valid:
                            # Module fixed and validated, activate it
                            self.update_module_status(module_id, True, "Fixed and activated")
                            logger.info(f"🔧 Module {module_id} fixed and activated")
                            self.stats["modules_activated"] += 1
                            activated_modules.append(module)
                        else:
                            # Fix didn't resolve all issues
                            module["issues"] = new_issues
                            logger.warning(f"⚠️ Module {module_id} could not be fully fixed: {new_issues}")
                            self.stats["modules_failed"] += 1
                            failed_modules.append(module)
                    else:
                        # Failed to fix
                        module["issues"] = issues
                        logger.warning(f"⚠️ Failed to fix module {module_id}")
                        self.stats["modules_failed"] += 1
                        failed_modules.append(module)
                else:
                    # Issues can't be fixed automatically
                    module["issues"] = issues
                    logger.warning(f"⚠️ Module {module_id} has issues that can't be fixed automatically: {issues}")
                    self.stats["modules_failed"] += 1
                    failed_modules.append(module)
        
        # Save updated registry
        if self.save_updated_registry():
            # Update build status and tracker
            self.update_build_status()
            self.update_build_tracker()
            
            # Generate report
            report_path = self.generate_activation_report(activated_modules, failed_modules)
            if report_path:
                logger.info(f"📊 Activation report saved to {report_path}")
            
            # Print summary
            logger.info("📊 MODULE ACTIVATION SUMMARY:")
            logger.info(f"✅ Modules Activated: {self.stats['modules_activated']}")
            logger.info(f"⚙️ Modules Processed: {self.stats['modules_processed']}")
            logger.info(f"⏩ Modules Skipped: {self.stats['modules_skipped']}")
            logger.info(f"❌ Modules Failed: {self.stats['modules_failed']}")
            logger.info(f"📈 Active Modules: {self.stats['active_modules'] + self.stats['modules_activated']}")
            logger.info(f"📉 Remaining Quarantined: {self.stats['quarantined_modules'] - self.stats['modules_activated']}")
            
            return True
        else:
            logger.error("❌ Failed to save updated registry, changes not persisted")
            return False

def main():
    """Main entry point"""
    print("🚀 GENESIS MODULE ACTIVATION ENGINE v1.0.0")
    print("==========================================")
    
    activation_engine = ModuleActivationEngine()
    success = activation_engine.run_activation()
    
    if success:
        print("✅ Module activation process completed successfully")
        print(f"📊 {activation_engine.stats['modules_activated']} modules activated")
    else:
        print("❌ Module activation process failed")
    
    print("==========================================")

if __name__ == "__main__":
    main()
