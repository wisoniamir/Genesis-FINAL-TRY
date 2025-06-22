#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 GENESIS DUPLICATE VERIFICATION ENGINE v7.0.0
ARCHITECT MODE v7.0.0 - ZERO TOLERANCE DUPLICATE ELIMINATION

🎯 CORE MISSION:
Verify and eliminate ALL duplicates before any upgrades
Enforce ARCHITECT MODE directive: DUPLICATION = FAILURE

🛡️ ZERO TOLERANCE ENFORCEMENT:
- NO DUPLICATES: Scan and eliminate all _backup, _copy, _v2 files
- NO ISOLATION: Verify no duplicate logic exists
- NO VIOLATIONS: Complete compliance verification
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GENESIS_DUPLICATE_SCANNER")

class GenesisDuplicateVerificationEngine:
    """Zero tolerance duplicate detection and elimination engine"""
    
    def __init__(self):
        self.workspace = Path("c:/Users/patra/Genesis FINAL TRY")
        self.duplicates_found = []
        self.duplicates_removed = 0
        self.violations_detected = 0
        
    def scan_for_duplicates(self) -> List[Dict]:
        """Scan entire workspace for duplicate files"""
        logger.info("🔍 Scanning for duplicate files...")
        
        duplicates = []
        duplicate_patterns = [
            '.backup',
            '.wiring_backup', 
            '_backup',
            '_copy',
            '_v2',
            '_duplicate',
            '_temp',
            '_recovered_1',
            '_recovered_2',
            '.QUARANTINED'
        ]
        
        # Scan all Python files
        for py_file in self.workspace.rglob("*.py"):
            file_name = py_file.name
            file_path = str(py_file)
            
            # Check for duplicate patterns
            for pattern in duplicate_patterns:
                if pattern in file_name:
                    duplicates.append({
                        'file': file_path,
                        'name': file_name,
                        'pattern': pattern,
                        'type': 'duplicate_file',
                        'size': py_file.stat().st_size if py_file.exists() else 0
                    })
                    break
        
        # Find base file duplicates (same name, different locations)
        base_files = {}
        for py_file in self.workspace.rglob("*.py"):
            if not any(pattern in py_file.name for pattern in duplicate_patterns):
                base_name = py_file.name
                if base_name not in base_files:
                    base_files[base_name] = []
                base_files[base_name].append(py_file)
        
        # Identify multiple instances of same file
        for base_name, file_list in base_files.items():
            if len(file_list) > 1:
                # Sort by path complexity - keep the simplest path
                file_list.sort(key=lambda x: (len(str(x).split('/')), str(x).count('QUARANTINE')))
                for duplicate_file in file_list[1:]:  # Mark all but first as duplicates
                    duplicates.append({
                        'file': str(duplicate_file),
                        'name': duplicate_file.name,
                        'pattern': 'multiple_instances',
                        'type': 'location_duplicate',
                        'size': duplicate_file.stat().st_size
                    })
        
        self.duplicates_found = duplicates
        logger.info(f"🔍 Found {len(duplicates)} duplicate files")
        return duplicates
    
    def verify_genesis_desktop_state(self) -> Dict:
        """Verify current state of genesis_desktop files"""
        logger.info("🔍 Verifying genesis_desktop file state...")
        
        desktop_files = []
        for file_path in self.workspace.rglob("*desktop*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.json']:
                desktop_files.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'is_backup': any(pattern in file_path.name for pattern in ['.backup', '_backup', '.wiring_backup'])
                })
        
        # Identify primary genesis_desktop.py
        primary_desktop = None
        backup_desktops = []
        
        for file_info in desktop_files:
            if file_info['name'] == 'genesis_desktop.py' and not file_info['is_backup']:
                if 'QUARANTINE' not in file_info['path']:
                    primary_desktop = file_info
            elif 'genesis_desktop' in file_info['name'] and file_info['is_backup']:
                backup_desktops.append(file_info)
        
        return {
            'primary_desktop': primary_desktop,
            'backup_desktops': backup_desktops,
            'all_desktop_files': desktop_files,
            'duplicates_count': len(backup_desktops)
        }
    
    def eliminate_duplicates(self, duplicates: List[Dict]):
        """Eliminate detected duplicate files"""
        logger.info(f"🗑️ Eliminating {len(duplicates)} duplicate files...")
        
        for duplicate in duplicates:
            file_path = Path(duplicate['file'])
            try:
                if file_path.exists():
                    # Move to quarantine instead of deleting
                    quarantine_dir = self.workspace / "DUPLICATE_QUARANTINE"
                    quarantine_dir.mkdir(exist_ok=True)
                    
                    quarantine_path = quarantine_dir / file_path.name
                    counter = 1
                    while quarantine_path.exists():
                        name_parts = file_path.name.split('.')
                        quarantine_path = quarantine_dir / f"{name_parts[0]}_dup_{counter}.{name_parts[1]}"
                        counter += 1
                    
                    file_path.rename(quarantine_path)
                    self.duplicates_removed += 1
                    logger.info(f"  🗑️ Quarantined: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed to quarantine {file_path.name}: {e}")
                self.violations_detected += 1
    
    def verify_no_duplicates_remain(self) -> bool:
        """Final verification that no duplicates remain"""
        logger.info("🔍 Final verification - scanning for remaining duplicates...")
        
        remaining_duplicates = self.scan_for_duplicates()
        
        # Filter out quarantined items
        active_duplicates = [
            dup for dup in remaining_duplicates 
            if 'QUARANTINE' not in dup['file']
        ]
        
        if len(active_duplicates) == 0:
            logger.info("✅ No duplicates remain - system clean")
            return True
        else:
            logger.warning(f"⚠️ {len(active_duplicates)} duplicates still detected:")
            for dup in active_duplicates:
                logger.warning(f"  - {dup['name']} ({dup['pattern']})")
            return False
    
    def generate_verification_report(self):
        """Generate duplicate verification report"""
        desktop_state = self.verify_genesis_desktop_state()
        
        report_content = f"""# 🔍 GENESIS DUPLICATE VERIFICATION REPORT v7.0.0
## ARCHITECT MODE v7.0.0 - ZERO TOLERANCE ENFORCEMENT

**Verification Date:** {datetime.now().isoformat()}
**Engine:** GENESIS Duplicate Verification Engine v7.0.0

---

## ✅ DUPLICATE SCAN RESULTS

### 📊 **ELIMINATION SUMMARY**
- **Duplicates Found:** {len(self.duplicates_found)}
- **Duplicates Removed:** {self.duplicates_removed}
- **Violations Detected:** {self.violations_detected}
- **System Status:** {'CLEAN' if self.duplicates_removed == len(self.duplicates_found) else 'VIOLATIONS REMAIN'}

### 🖥️ **GENESIS DESKTOP FILE STATE**
- **Primary Desktop File:** {'✅ Found' if desktop_state['primary_desktop'] else '❌ Missing'}
- **Backup Files Found:** {len(desktop_state['backup_desktops'])}
- **Total Desktop Files:** {len(desktop_state['all_desktop_files'])}

#### **Primary Desktop File Details:**
{f"- Path: {desktop_state['primary_desktop']['path']}" if desktop_state['primary_desktop'] else "- ❌ No primary genesis_desktop.py found"}
{f"- Size: {desktop_state['primary_desktop']['size']} bytes" if desktop_state['primary_desktop'] else ""}

#### **Backup Files Eliminated:**
{chr(10).join(f"- {backup['name']} ({backup['size']} bytes)" for backup in desktop_state['backup_desktops'])}

### 🔍 **DUPLICATE PATTERNS DETECTED**
{chr(10).join(f"- {dup['pattern']}: {dup['name']}" for dup in self.duplicates_found)}

### ✅ **ARCHITECT MODE COMPLIANCE**
- **NO DUPLICATES:** {'✅ ENFORCED' if self.duplicates_removed == len(self.duplicates_found) else '❌ VIOLATIONS REMAIN'}
- **ZERO TOLERANCE:** {'✅ ACTIVE' if self.violations_detected == 0 else '❌ VIOLATIONS DETECTED'}
- **SYSTEM CLEAN:** {'✅ VERIFIED' if self.verify_no_duplicates_remain() else '❌ DUPLICATES REMAIN'}

### 🎯 **UPGRADE READINESS**
{'✅ SYSTEM READY FOR DASHBOARD UPGRADE' if self.duplicates_removed == len(self.duplicates_found) else '❌ MUST ELIMINATE DUPLICATES BEFORE UPGRADE'}

---

*Report generated by GENESIS Duplicate Verification Engine v7.0.0*
*ARCHITECT MODE v7.0.0 Zero Tolerance Enforcement*
"""
        
        # Save verification report
        try:
            with open(self.workspace / "GENESIS_DUPLICATE_VERIFICATION_REPORT.md", 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info("✅ Generated duplicate verification report")
        except Exception as e:
            logger.error(f"❌ Failed to save verification report: {e}")
    
    def run_duplicate_verification(self) -> bool:
        """Execute complete duplicate verification and elimination"""
        logger.info("🚀 Starting GENESIS duplicate verification...")
        
        # Phase 1: Scan for duplicates
        logger.info("📊 Phase 1: Scanning for duplicates")
        duplicates = self.scan_for_duplicates()
        
        if len(duplicates) == 0:
            logger.info("✅ No duplicates found - system clean")
            self.generate_verification_report()
            return True
        
        # Phase 2: Eliminate duplicates
        logger.info("🗑️ Phase 2: Eliminating duplicates")
        self.eliminate_duplicates(duplicates)
        
        # Phase 3: Final verification
        logger.info("🔍 Phase 3: Final verification")
        clean_system = self.verify_no_duplicates_remain()
        
        # Phase 4: Generate report
        logger.info("📝 Phase 4: Generating report")
        self.generate_verification_report()
        
        if clean_system:
            logger.info("🎉 DUPLICATE VERIFICATION COMPLETE!")
            logger.info(f"📊 Summary: {self.duplicates_removed} duplicates eliminated")
            logger.info("✅ System clean - ready for dashboard upgrade")
            return True
        else:
            logger.error("❌ DUPLICATE VERIFICATION FAILED!")
            logger.error("🚨 Violations remain - upgrade blocked")
            return False

if __name__ == "__main__":
    engine = GenesisDuplicateVerificationEngine()
    success = engine.run_duplicate_verification()
    
    if success:
        print("\n🏆 DUPLICATE VERIFICATION SUCCESSFUL!")
        print("✅ System clean - ready for dashboard upgrade")
    else:
        print("\n❌ DUPLICATE VERIFICATION FAILED!")
        print("🚨 Must eliminate all duplicates before proceeding")
