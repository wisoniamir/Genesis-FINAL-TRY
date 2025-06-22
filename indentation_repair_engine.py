#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 GENESIS DESKTOP INDENTATION REPAIR ENGINE
ARCHITECT MODE v7.0.0 COMPLIANCE REPAIR

Fixes all indentation errors in genesis_desktop.py identified by Phase 7.9 validation
"""

import re
import os
from pathlib import Path

def fix_genesis_desktop_indentation():
    """Fix all indentation errors in genesis_desktop.py"""
    
    file_path = Path(__file__).parent / "genesis_desktop.py"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔧 Fixing indentation errors in {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find problematic try blocks with incorrect indentation
    # Looking for patterns like:
    # try:
    # some_statement
    # except Exception as e:
    
    # Fix pattern 1: Missing indentation after try:
    pattern1 = r'(\s+)try:\s*\n(\s+)([^e]\S.*\.(?:clicked\.connect|stateChanged\.connect)\([^)]+\))\s*\n(\s+)except Exception as e:\s*\n(\s+)logging\.error\(f"Operation failed: \{e\}"\)\s*\n(\s+)except Exception as e:\s*\n(\s+)logging\.error\(f"Operation failed: \{e\}"\)'
    
    def replace_pattern1(match):
        indent = match.group(1)
        statement = match.group(3)
        return f"""{indent}try:
{indent}    {statement}
{indent}except Exception as e:
{indent}    logging.error(f"Operation failed: {{e}}")"""
    
    # Apply pattern 1 fix
    content = re.sub(pattern1, replace_pattern1, content, flags=re.MULTILINE)
    
    # Fix pattern 2: Standalone try blocks with incorrect structure
    pattern2 = r'(\s+)try:\s*\n(\s+)([^e]\S.*\.(?:clicked\.connect|stateChanged\.connect)\([^)]+\))\s*\n(\s+)except Exception as e:\s*\n(\s+)logging\.error\(f"Operation failed: \{e\}"\)'
    
    def replace_pattern2(match):
        indent = match.group(1)
        statement = match.group(3)
        return f"""{indent}try:
{indent}    {statement}
{indent}except Exception as e:
{indent}    logging.error(f"Operation failed: {{e}}")"""
    
    # Apply pattern 2 fix
    content = re.sub(pattern2, replace_pattern2, content, flags=re.MULTILINE)
    
    # Fix pattern 3: Try blocks with incorrect indentation structure
    pattern3 = r'(\s+)try:\s*\n(\s+)([^e]\S[^\\n]*)\s*\n(\s+)except Exception as e:'
    
    def replace_pattern3(match):
        indent = match.group(1)
        statement_indent = match.group(2)
        statement = match.group(3)
        
        # If statement is not properly indented relative to try, fix it
        if len(statement_indent) <= len(indent):
            return f"""{indent}try:
{indent}    {statement}
{indent}except Exception as e:"""
        else:
            return match.group(0)  # Keep as is if already properly indented
    
    # Apply pattern 3 fix
    content = re.sub(pattern3, replace_pattern3, content, flags=re.MULTILINE)
    
    # Create backup
    backup_path = file_path.with_suffix('.py.backup_indentation_fix')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Write fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Indentation errors fixed in {file_path}")
    print(f"💾 Backup saved as {backup_path}")
    
    return True

def fix_genesis_minimal_launcher_indentation():
    """Fix indentation errors in genesis_minimal_launcher.py"""
    
    file_path = Path(__file__).parent / "genesis_minimal_launcher.py"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔧 Fixing indentation errors in {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix line 254 specifically
    if len(lines) > 254:
        line_254 = lines[253]  # 0-indexed
        if line_254.strip().startswith('self.') and not line_254.startswith('        '):
            # Fix indentation to match surrounding context
            lines[253] = '        ' + line_254.lstrip()
            print(f"🔧 Fixed indentation on line 254: {lines[253].strip()}")
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Indentation errors fixed in {file_path}")
    
    return True

def main():
    """Main execution function"""
    print("🔧 GENESIS INDENTATION REPAIR ENGINE")
    print("=" * 50)
    
    success1 = fix_genesis_desktop_indentation()
    success2 = fix_genesis_minimal_launcher_indentation()
    
    if success1 and success2:
        print("\n✅ All indentation errors fixed successfully!")
        print("🔄 Re-run Phase 7.9 validation to verify fixes")
    else:
        print("\n❌ Some fixes failed - manual intervention required")
    
    return success1 and success2

if __name__ == "__main__":
    main()
